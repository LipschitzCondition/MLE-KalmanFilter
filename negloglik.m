function nll = negloglik(params, y, Sigma_w, n, p, k)
    % Reshape parameter vector into A, B, C, D
    offset = 0;
    A = reshape(params(offset + 1 : offset + n*n), n, n); offset = offset + n*n; % Extract the first n*n elements, move the counting forward
    B = reshape(params(offset + 1 : offset + n*k), n, k); offset = offset + n*k; % Extract the first n*k elements, move the counting forward
    C = reshape(params(offset + 1 : offset + p*n), p, n); offset = offset + p*n; % Extract the first p*n elements, move the counting forward
    D = reshape(params(offset + 1 : offset + p*k), p, k);

    % Setting para
    T = size(y, 2);
    x_post_prev = zeros(n, 1);        % x_{1|0}
    P_post_prev = 1e-3 * eye(n);      % P_{1|0}

    nll = 0;  % Negative log-likelihood

    for t = 1:T
        % === Prediction ===
        x_prior = A * x_post_prev;
        P_prior = A * P_post_prev * A' + B * Sigma_w * B';

        % === Innovation ===
        S = C * P_prior * C' + D * Sigma_w * D';
        a = y(:,t) - C * x_prior;

        % === Add log-likelihood contribution ===
        nll = nll + 0.5 * (log(det(S)) + a' / S * a);

        % === Update ===
        K = (A * P_post_prev * C' + B * Sigma_w * D') / S;
        x_post = x_prior + K * a;
        P_post = P_prior - K * S * K';

        % Prepare for next round
        x_post_prev = x_post;
        P_post_prev = P_post;
    end
end