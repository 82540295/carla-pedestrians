function kb_s2_se3_check(inDir)
%
%
%
%   se3_gt.txt se3_pres2.txt se3_final.txt se3_ekf.txt se3_events.txt
%

if nargin < 1 || isempty(inDir)
    inDir = '/tmp';
end
inDir = char(inDir);

gt     = dlmread(fullfile(inDir, 'se3_gt.txt'), '');
preS2  = dlmread(fullfile(inDir, 'se3_pres2.txt'), '');
finalT = dlmread(fullfile(inDir, 'se3_final.txt'), '');
ekf    = dlmread(fullfile(inDir, 'se3_ekf.txt'), '');
evs    = dlmread(fullfile(inDir, 'se3_events.txt'), '');

L = sum(sqrt(sum(diff(gt, 1, 1).^2, 2)));
fprintf('input    : %s\n', inDir);
fprintf('sizes    : gt=%dx%d pre=%dx%d final=%dx%d ekf=%dx%d events=%dx%d\n', ...
        size(gt,1),size(gt,2), size(preS2,1),size(preS2,2), size(finalT,1),size(finalT,2), ...
        size(ekf,1),size(ekf,2), size(evs,1),size(evs,2));
fprintf('gt length: %.1f m   frames: %d   events: %d\n', L, size(gt,1), size(evs,1));

ok = true;
ok = report('preS2 (A only) ', preS2,  gt, NaN,    NaN)   && ok;
ok = report('S2    (A+loop)', finalT, gt, 55.92,  87.25)  && ok;
ok = report('EKF   (core)  ', ekf,    gt, 842.54, 477.76) && ok;

fprintf('ratio EKF/S2 terminal : %.2fx   (paper: 15x)\n', ...
        anchored100_terminal(ekf, gt) / anchored100_terminal(finalT, gt));
fprintf('scale2d rmse  S2/EKF : %.2f / %.2f  (diagnostic only)\n', ...
        scale2d_rmse(finalT, gt), scale2d_rmse(ekf, gt));

[se3, napp] = reanchor_se3(preS2, evs, 250);
fprintf('SE(3) variant        : terminal=%.2f rmse=%.2f n_app=%d/%d  (no paper reference)\n', ...
        anchored100_terminal(se3, gt), anchored100(se3, gt), napp, size(evs,1));

if ok
    fprintf('\nRESULT: PASS  (all paper-reported numbers reproduced within 0.5 m)\n');
else
    fprintf('\nRESULT: FAIL  (see mismatch deltas above)\n');
end
end

% ────────────────────────────────────────────────────────────────────────
function ok = report(tag, est, gt, expTerm, expRmse)
    term = anchored100_terminal(est, gt);
    rmse = anchored100(est, gt);
    if isnan(expTerm)
        fprintf('%s : terminal=%9.2f  rmse=%9.2f  (no reference)\n', tag, term, rmse);
        ok = true;
        return;
    end
    dt = term - expTerm;
    dr = rmse - expRmse;
    pass = abs(dt) <= 0.5 && abs(dr) <= 0.5;
    ok = pass;
    if pass, verdict = 'PASS'; else, verdict = 'FAIL'; end
    fprintf('%s : terminal=%9.2f (exp %7.2f, d=%+6.2f)  rmse=%9.2f (exp %7.2f, d=%+6.2f)  %s\n', ...
            tag, term, expTerm, dt, rmse, expRmse, dr, verdict);
end

function [ea, T, s] = anchored_fit(est, gt, k)
    n = min(size(est,1), size(gt,1)); est = est(1:n,:); gt = gt(1:n,:);
    e0 = est(1:k,:); g0 = gt(1:k,:);
    e0c = e0 - mean(e0,1); g0c = g0 - mean(g0,1);
    [U,S,Vt] = svd(g0c' * e0c, 'econ');
    d = det(Vt' * U');
    T = Vt' * diag([1 1 d]) * U';
    s = sum(S * [1; 1; d]) / sum(e0c(:).^2);
    tvec = mean(g0,1) - s * (mean(e0,1) * T);
    ea = s * (est * T) + tvec;
end

function rm = anchored100(est, gt)
    ea = anchored_fit(est, gt, 100);
    g  = gt(1:size(ea,1),:);
    rm = sqrt(mean(sum((ea - g).^2, 2)));
end

function te = anchored100_terminal(est, gt)
    ea = anchored_fit(est, gt, 100);
    g  = gt(1:size(ea,1),:);
    te = sqrt(sum((ea(end,:) - g(end,:)).^2, 2));
end

function rm = scale2d_rmse(est, gt)
    n = min(size(est,1), size(gt,1)); est = est(1:n,:); gt = gt(1:n,:);
    e = est - est(1,:); g = gt - gt(1,:);
    Le = sum(sqrt(sum(diff(e,1,1).^2, 2)));
    Lg = sum(sqrt(sum(diff(g,1,1).^2, 2)));
    s = Lg / Le;
    ea = e * s;
    rm = sqrt(mean(sum((ea(:,1:2) - g(:,1:2)).^2, 2)));
end

function [traj, napp] = reanchor_se3(traj, events, win)
    n = size(traj,1); base = traj;
    d0 = sqrt(sum((base - base(1,:)).^2, 2));
    sra = find(d0 > 1, 1, 'first');
    if isempty(sra), sra = 1; end
    aWin = base(sra:min(sra+win-1, n), :);
    ca = mean(aWin, 1);
    evs = sortrows(events, 1);
    napp = 0;
    for k = 1:size(evs,1)
        t_now = evs(k,1);
        if k < size(evs,1), t_next = evs(k+1,1); else, t_next = n+1; end
        cWin = base(max(t_now-win+1, 1):min(t_now, t_next-1), :);
        seg  = (t_now+1):(t_next-1);
        if numel(cWin) < 30 || numel(seg) < 2 || size(aWin,1) ~= size(cWin,1)
            continue;
        end
        [~, ~, T] = procrustes(aWin, cWin, 'Scaling', false);
        Rmat = T.T;
        cc = mean(cWin, 1);
        tvec = ca - (cc * Rmat);
        traj(seg,:) = (base(seg,:) * Rmat) + tvec;
        napp = napp + 1;
    end
end
