function nll = negloglik(params, y, Sigma_w, n, p, k)
    % Reshape para vector into A, B, C, D
    offset = 0;
    A = reshape(params(offset + 1 : offset + n*n), n, n); offset = offset + n*n; % Extract the first n*n elements, move the counting forward
    B = reshape(params(offset + 1 : offset + n*k), n, k); offset = offset + n*k; % Extract the first n*k elements, move the counting forward
    C = reshape(params(offset + 1 : offset + p*n), p, n); offset = offset + p*n; % Extract the first p*n elements, move the counting forward
    D = reshape(params(offset + 1 : offset + p*k), p, k);

    % Setting para
    T = size(y, 2);

    % Initial prior belief about x_{1|0}'s distribution:
    x_post_prev = zeros(n, 1);        % hatx_{1|0}
    P_post_prev = eye(n);      % Sigma_{1|0}

    nll = 0;  % Negative log-likelihood

    for t = 1:T

        % === Prediction ===

        % mean for x_{t+1|t-1}
        x_prior = A * x_post_prev;                          % A hatx_{t|t-1}
        
        % var for x_{t+1|t-1}
        P_prior = A * P_post_prev * A' + B * Sigma_w * B';  % A Sigma_{t|t-1} A' + BB'


        % === Innovation ===
        % var for y_{t|t-1}
        S = C * P_prior * C' + D * Sigma_w * D';            % C Sigma_{t|t-1} C' + DD'
        
        % news
        a = y(:,t) - C * x_prior;                           % y_{t|t-1} - C hatx_{t|t-1}

        % === Add log-likelihood contribution ===
        nll = nll + 0.5 * (log(det(S)) + a' / S * a);

        % === Update ===
        % Kalman gain
        K = (A * P_post_prev * C' + B * Sigma_w * D') / S;

        % Posterior belief for x_{t+1|t}
        % mean
        x_post = x_prior + K * a;
        % var
        P_post = P_prior - K * S * K';

        % Prepare for next round: Posterior serves as prior
        x_post_prev = x_post;
        P_post_prev = P_post;
    end
end
