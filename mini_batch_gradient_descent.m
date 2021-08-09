function [theta_0, theta, J_history] = mini_batch_gradient_descent(x, y, theta, alpha, iterations)

theta_0 = [];
    m = size(y);
%     J_history = zeros(iterations, 1);
    j = 2;
    for i = 1:iterations
         t = randperm(size(x,1),m(1));
%        if alpha*(1/m(1))*sum(h-y) > 0.001
         for j = 1:j
            h = theta(1)*x(t(j),1) + theta(2)*x(t(j),2) + theta(3)*x(t(j),3);
            t1 = theta(1) - alpha*(1/m(1))*sum(h-y);
            t2 = theta(2) - alpha*(1/m(1))*sum(h-y).*x(t(j),2);
            t3 = theta(3) - alpha*(1/m(1))*sum(h-y).*x(t(j),3);
            theta(1) = t1;
            theta(2) = t2;
            theta(3) = t3;
%        end
%        else
%           break
         end
         theta_0 = [theta_0 theta];
         J_history(i,1) = compute_cost(x,y,theta);
    end 
end