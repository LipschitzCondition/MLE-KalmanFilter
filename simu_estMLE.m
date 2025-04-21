% Dimensions of the space
T = 1000;        % time horizons
n = 2;          % state dimension
p = 2;          % observation dimension
k = 2;          % shock dimension (w_t)

% True parameters, used to generate y^T
A_true = [0.8 0.1; 0.2 0.9];
B_true = eye(n);
C_true = [1 0; 0 1];
D_true = eye(p);

Sigma_w = eye(k);  % shock covariance, assumed by us to be I


% Initialize
x = zeros(n, T+1);
y = zeros(p, T);

% Generate stochastic shocks and simulate sequences
rng(229); % set seed
w = mvnrnd(zeros(k,1), Sigma_w, T)';

for t = 1:T
    x(:,t+1) = A_true * x(:,t) + B_true * w(:,t);
    y(:,t)   = C_true * x(:,t) + D_true * w(:,t);
end


% === STEP 2: MLE for A, B, C, D ===
% Initial guess for MLE estimation of parameters
params0 = [A_true(:); B_true(:); C_true(:); D_true(:)] + 0.1*randn(n^2 + n*k + p*n + p*k, 1);

% Objective
objfun = @(params) negloglik(params, y, Sigma_w, n, p, k); % With 'params' unfilled, to do optimization

% Optimization
% options = optimset('Display', 'off', 'MaxFunEvals', 1e5, 'MaxIter', 1e5);
% est_params = fminsearch(objfun, params0, options);
est_params = fminsearch(objfun, params0);


% Extract estimated A, B, C, D
offset = 0;
A_hat = reshape(est_params(offset + 1 : offset + n*n), n, n); offset = offset + n*n; % Extract the first n*n elements, move the counting forward
B_hat = reshape(est_params(offset + 1 : offset + n*k), n, k); offset = offset + n*k; % Extract the first n*k elements, move the counting forward
C_hat = reshape(est_params(offset + 1 : offset + p*n), p, n); offset = offset + p*n; % Extract the first p*n elements, move the counting forward
D_hat = reshape(est_params(offset + 1 : offset + p*k), p, k);

disp('Estimated A:'); disp(A_hat);
disp('Estimated B:'); disp(B_hat);
disp('Estimated C:'); disp(C_hat);
disp('Estimated D:'); disp(D_hat);




% === STEP 3: Build state-space model in MATLAB format ===
% The system:
%   x_{t+1} = A x_t + B w_t
%   y_t     = C x_t + D w_t
% Matches MATLAB's built-in Kalman function:
%   x[k+1] = A x[k] + B u[k] + G w[k]
%   y[k]   = C x[k] + D u[k] + H w[k]


% Create state-space system with dummy B = D = 0
sys = ss(A_hat, B_hat, C_hat, D_hat, 1);  % discrete-time system, Ts=1

Q = eye(k);                     % because w_t ~ N(0, I)
R = zeros(p);                   % no additional measurement noise
N = zeros(n, p);                % no process-measurement noise cross-correlation


% === STEP 4: Apply built-in Kalman filter ===
[kest, L, P] = kalman(sys, Q, R, N);



tvec = (1:T)';
x0 = zeros(size(kest.A, 1), 1);       % match size of estimator states
x_est = lsim(kest, y', tvec, x0);     % y' is [T × p]

% === Extract true states (excluding x_0) for plotting ===
x_true_plot = x(:, 2:end)';  % [T × n]

% === Plot estimated vs. true states ===
n_plot = size(x_true_plot, 2);  % number of true state variables

figure;
for i = 1:n_plot
    subplot(n_plot, 1, i);
    plot(1:T, x_true_plot(:, i), 'k-', 'LineWidth', 1.5); hold on;
    plot(1:T, x_est(:, i), 'r--', 'LineWidth', 1.5);
    legend('True x_t', 'Estimated x_t');
    xlabel('Time'); ylabel(['x_', num2str(i)]);
    title(['State estimate for x_', num2str(i)]);
end




%{
% Initial guess for MLE estimation of parameters
params0 = [A_true(:); B_true(:); C_true(:); D_true(:)] + 0.1*randn(n^2 + n*k + p*n + p*k, 1);

% Objective
objfun = @(params) negloglik(params, y, Sigma_w, n, p, k); % With 'params' unfilled, to do optimization

% Optimization
est_params = fminsearch(objfun, params0);

% Extract estimated A, B, C, D
offset = 0;
A_hat = reshape(est_params(offset + 1 : offset + n*n), n, n); offset = offset + n*n; % Extract the first n*n elements, move the counting forward
B_hat = reshape(est_params(offset + 1 : offset + n*k), n, k); offset = offset + n*k; % Extract the first n*k elements, move the counting forward
C_hat = reshape(est_params(offset + 1 : offset + p*n), p, n); offset = offset + p*n; % Extract the first p*n elements, move the counting forward
D_hat = reshape(est_params(offset + 1 : offset + p*k), p, k);

disp('Estimated A:'); disp(A_hat);
disp('Estimated B:'); disp(B_hat);
disp('Estimated C:'); disp(C_hat);
disp('Estimated D:'); disp(D_hat);


% Use estimated A, B, C, D to do Kalman filtering, with Sigma_w assumed initially
[x_filtered, P_filtered] = kalman_filter(y, A_hat, B_hat, C_hat, D_hat, Sigma_w);


% Plotting, not necessarily
figure;
for i = 1:size(x,1)
    subplot(size(x,1),1,i);
    plot(1:T, x(i,2:end), 'k-', 'LineWidth', 1.5); hold on;
    plot(1:T, x_filtered(i,:), 'r--');
    legend('True x_t', 'Estimated x_t');
    title(['State var x_', num2str(i)]);
end
%}
