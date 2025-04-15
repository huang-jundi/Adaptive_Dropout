function y = Infill_ESSI(x,kriging_model,fmin,best_x,optimize_dim)
n = size(x,1);
new_x = repmat(best_x,n,1);
new_x(:,optimize_dim) = x;

[u,s] = GP_predict(new_x,kriging_model);
y = (fmin-u).*normcdf((fmin-u)./s)+s.*normpdf((fmin-u)./s);
end
