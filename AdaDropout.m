clearvars;clc;close all;
% objective function
fun_name = 'Ellipsoid';
% number of variables
num_vari = 100;
% lower and upper bounds
lower_bound = -5.12*ones(1,num_vari);
upper_bound = 5.12*ones(1,num_vari);
% number of initial design points
num_initial = 2*num_vari;
% number of maximum evaluations
max_evaluation = 10*num_vari;
% initial design
sample_x = lhsdesign(num_initial,num_vari,'criterion','maximin','iterations',1000).*(upper_bound-lower_bound)+lower_bound;
sample_y = feval(fun_name,sample_x);
iteration = 1;
evaluation =  size(sample_x,1);
[fmin,ind] = min(sample_y);
best_x = sample_x(ind,:);
fmin_record(iteration,1) = fmin;
fprintf('AdaDropout on %d-D %s, iteration: %d, evaluation: %d, best: %0.4g\n',num_vari,fun_name,iteration-1,evaluation,fmin);
dim=num_vari;
while evaluation < max_evaluation
    % train GP models
    GP_model = GP_train(sample_x,sample_y,lower_bound,upper_bound,1,0.01,100);
    % select optimizing variables
    optimize_dim=randperm(100,dim);  
    Lower_bound=lower_bound(optimize_dim);
    Upper_bound=upper_bound(optimize_dim);
    pop_size = max(10,4*dim);
    max_gen = 200*dim/pop_size;
    [candidate_x,ESSI] = Optimizer_GA(@(x)-Infill_ESSI(x,GP_model,fmin,best_x,optimize_dim),dim,Lower_bound,Upper_bound,pop_size,max_gen);
    % get a new solution
    infill_x = best_x;
    infill_x(:,optimize_dim) = candidate_x;
    % evaluate the new solution
    infill_y = feval(fun_name,infill_x);
    % determine whether to dropout variable
    if infill_y>fmin
        if dim>1
            dim=dim-1;
        end
    end
    % update dataset
    iteration = iteration + 1;
    sample_x = [sample_x;infill_x];
    sample_y = [sample_y;infill_y];
    [fmin,ind] = min(sample_y);
    best_x = sample_x(ind,:);
    fmin_record(iteration,1) = fmin;
    evaluation = evaluation + size(infill_x,1);
    fprintf('AdaDropout on %d-D %s, iteration: %d, evaluation: %d, best: %0.4g\n',num_vari,fun_name,iteration-1,evaluation,fmin);
end
