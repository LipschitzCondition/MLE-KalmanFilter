function [x_prior, x_post, P_prior, P_post] = kalman_filter_strict(y, A, B, C, D, Sigma_w)
    [p, T] = size(y);
    n = size(A, 1);
    k = size(B, 2);

    % Allocate space
    x_prior = zeros(n, T);
    x_post  = zeros(n, T);
    P_prior = zeros(n, n, T);
    P_post  = zeros(n, n, T);

    % Initial condition: x_1|0 = 0, P_1|0 = small nonzero matrix
    x_post_prev = zeros(n, 1);      % x_{1|0}
    P_post_prev = 1e-3 * eye(n);    % P_{1|0}

    for t = 1:T
        % === Prediction step ===
        x_prior(:,t) = A * x_post_prev;
        P_prior(:,:,t) = A * P_post_prev * A' + B * Sigma_w * B';

        % Innovation
        S = C * P_prior(:,:,t) * C' + D * Sigma_w * D';
        e = y(:,t) - C * x_prior(:,t);

        % Kalman gain
        K = (A * P_post_prev * C' + B * Sigma_w * D') / S;

        % === Update step ===
        x_post(:,t) = x_prior(:,t) + K * e;
        P_post(:,:,t) = P_prior(:,:,t) - K * S * K';

        % Prepare for next step
        x_post_prev = x_post(:,t);
        P_post_prev = P_post(:,:,t);
    end
end
