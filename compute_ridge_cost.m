function J = compute_ridge_cost(x, y, theta, lambda)
    
    h = theta(1)*x(:,1) + theta(2)*x(:,2)+ theta(3)*x(:,3);
    
    J = (1/2)*(sum((h-y).^2)+lambda*sum(theta.^2));
end