function [theta_0, theta, J_history] = ridge_batch_gradient_descent(x, y, theta, alpha, iterations, lambda)

    theta_0 = [];
    m = size(y);
    J_history = zeros(iterations, 1);
    
    for i = 1:iterations
            h = theta(1)*x(:,1) + theta(2)*x(:,2) + theta(3)*x(:,3);
%             if alpha*(1/m(1))*sum(h-y) > 0.001
                t1 = (1-alpha*lambda)*theta(1) - alpha*(1/m(1))*sum(h-y);
                t2 = (1-alpha*lambda)*theta(2) - alpha*(1/m(1))*sum((h-y).*x(:,2));
                t3 = (1-alpha*lambda)*theta(3) - alpha*(1/m(1))*sum((h-y).*x(:,3));
                theta(1) = t1;
                theta(2) = t2;
                theta(3) = t3;
                theta_0 = [theta_0 theta];
                J_history(i,1) = compute_ridge_cost(x,y,theta, lambda);
%             else 
%                 break
%             end
    end 
end