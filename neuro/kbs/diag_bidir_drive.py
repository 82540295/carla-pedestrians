#!/usr/bin/env python3
import argparse
import math
import sys

sys.path.insert(0, os.path.expanduser('~/下载/carla/CARLA_0.9.16/PythonAPI/carla'))
import carla


def aslist(x):
    return x if isinstance(x, (list, tuple)) else ([x] if x is not None else [])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--port', type=int, default=2000)
    ap.add_argument('--start', type=str, default='173.1,2.0')
    ap.add_argument('--out-dist', type=float, default=110.0)
    ap.add_argument('--frames', type=int, default=400)
    a = ap.parse_args()

    c = carla.Client('localhost', a.port)
    c.set_timeout(30.0)
    world = c.get_world()
    try:
        s0 = world.get_settings()
        if s0.synchronous_mode:
            s0.synchronous_mode = False
            s0.fixed_delta_seconds = None
            world.apply_settings(s0)
            print('世界曾被冻结在同步模式, 已恢复异步', flush=True)
    except Exception:
        pass
    m = world.get_map()
    print(f'当前地图={m.name}', flush=True)
    tms = [a for a in world.get_actors() if 'trafficmanager' in a.type_id]
    for a in tms:
        try:
            a.destroy()
        except Exception:
            pass
    for act in world.get_actors():
        t = act.type_id
        if t.startswith('vehicle.') or t.startswith('sensor.') \
                or t.startswith('walker.') or t.startswith('traffic'):
            try:
                act.destroy()
            except Exception:
                pass
    world.tick()
    left = [a for a in world.get_actors()
            if a.type_id.startswith('vehicle.')]
    print(f'清理后残留车辆数={len(left)} (销毁TM {len(tms)}个)', flush=True)

    sx, sy = [float(v) for v in a.start.split(',')]
    wp0 = aslist(m.get_waypoint(carla.Location(sx, sy, 0.0),
                                project_to_road=True))[0]
    print(f'部署点=({wp0.transform.location.x:.2f},{wp0.transform.location.y:.2f},'
          f'{wp0.transform.location.z:.2f}) yaw={wp0.transform.rotation.yaw:.1f}° '
          f'lane_id={wp0.lane_id} road_id={wp0.road_id} lane宽={wp0.lane_width:.1f}',
          flush=True)

    out_wp = wp0
    for dstep in range(20, 400, 10):
        nxt = aslist(wp0.next(dstep))
        if not nxt or nxt[0].road_id != wp0.road_id:
            break
        out_wp = nxt[0]
        if out_wp.transform.location.distance(
                wp0.transform.location) >= a.out_dist:
            break
    out_dest = carla.Location(out_wp.transform.location.x,
                              out_wp.transform.location.y,
                              out_wp.transform.location.z)
    print(f'去程目标=({out_dest.x:.1f},{out_dest.y:.1f}) '
          f'距离={out_dest.distance(wp0.transform.location):.0f}m', flush=True)

    bpm = world.get_blueprint_library()
    vbp = bpm.find('vehicle.lincoln.mkz_2020')
    spawn_tf = carla.Transform(
        carla.Location(wp0.transform.location.x - 12.0,
                       wp0.transform.location.y,
                       wp0.transform.location.z),
        wp0.transform.rotation)
    car = world.try_spawn_actor(vbp, spawn_tf)
    if car is None:
        for sp in m.get_spawn_points()[:30]:
            car = world.try_spawn_actor(vbp, sp)
            if car is not None:
                print(f'备选出生点生成成功: {sp.location}', flush=True)
                break
    if car is None:
        raise RuntimeError('车辆生成失败: 无可用位置')
    tm = c.get_trafficmanager(port=8021)

    from agents.navigation.behavior_agent import BehaviorAgent
    agent = BehaviorAgent(car, behavior='cautious')
    agent.follow_speed_limits(False)
    try:
        agent.set_max_speed(12 / 3.6)
    except AttributeError:
        try:
            agent.set_target_speed(12 / 3.6)
        except AttributeError:
            agent._max_speed = 12 / 3.6

    fence = {'n': 0}
    def on_col(e):
        t = e.other_actor.type_id
        if 'fence' in t.lower():
            fence['n'] += 1
            print(f'  [FENCE] 累计={fence["n"]} intensity={e.intensity:.0f} '
                  f'位置=({car.get_transform().location.x:.1f},'
                  f'{car.get_transform().location.y:.1f})', flush=True)
    col = world.spawn_actor(bpm.find('sensor.other.collision'),
                            carla.Transform(), car)
    col.listen(on_col)

    agent.set_destination(out_dest)

    print('frame  x        y        z       yaw°   车速   fence  距目标', flush=True)
    import time as _time
    for f in range(a.frames):
        try:
            ctl = agent.run_step()
            ctl.manual_gear_shift = False
            ctl.steer = max(-0.5, min(0.5, ctl.steer))
        except Exception as e:
            print(f'[err] {str(e)[:60]}', flush=True)
            ctl = carla.VehicleControl(throttle=1.0)
        car.apply_control(ctl)
        if f % 5 == 0:
            t = car.get_transform()
            v = car.get_velocity()
            sp = math.sqrt(v.x**2 + v.y**2 + v.z**2)
            d = t.location.distance(out_dest)
            print(f'{f:>5}  {t.location.x:8.2f} {t.location.y:7.2f} '
                  f'{t.location.z:8.2f} {t.rotation.yaw:7.1f}  {sp:5.1f} '
                  f'{fence["n"]:>4}  {d:6.1f}', flush=True)
            _time.sleep(0.06)

    print(f'\n结束: fence={fence["n"]}, '
          f'最终=({car.get_transform().location.x:.1f},'
          f'{car.get_transform().location.y:.1f}), '
          f'距目标={car.get_transform().location.distance(out_dest):.1f}m',
          flush=True)
    car.destroy()
    col.destroy()


if __name__ == '__main__':
    main()
