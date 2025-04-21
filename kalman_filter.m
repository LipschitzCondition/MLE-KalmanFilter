function [x_prior, x_post, P_prior, P_post] = kalman_filter(y, A, B, C, D, Sigma_w)
    [p, T] = size(y);  % p: dim of y_t
    n = size(A, 1);    % n: dim of x_t
    k = size(B, 2);    % k: dim of w_t

    % Initialization
    x_prior = zeros(n, T);
    x_post  = zeros(n, T);
    P_prior = zeros(n, n, T);
    P_post  = zeros(n, n, T);

    % Prior Distribution for x_{1|0}
    x_post_prev = zeros(n, 1);      % hatx_{1|0}
    % P_post_prev = 1e-3 * eye(n);    % Sigma_{1|0}
    P_post_prev = eye(n);

    for t = 1:T

        % === Prediction ===
        % mean for x_{t+1|t-1}
        x_prior(:,t) = A * x_post_prev;
        % var for x_{t+1|t-1}
        P_prior(:,:,t) = A * P_post_prev * A' + B * Sigma_w * B';


        % === Innovation ===
        % var for y_{t|t-1}
        S = C * P_prior(:,:,t) * C' + D * Sigma_w * D';
        % news
        a = y(:,t) - C * x_prior(:,t);


        % === Update ===
        % Kalman gain
        K = (A * P_post_prev * C' + B * Sigma_w * D') / S;

        % Posterior belief for x_{t+1|t}
        % mean
        x_post(:,t) = x_prior(:,t) + K * a;
        % var
        P_post(:,:,t) = P_prior(:,:,t) - K * S * K';


        % === Prepare for next step ===
        x_post_prev = x_post(:,t);
        P_post_prev = P_post(:,:,t);
    end
end
