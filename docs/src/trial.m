
[X,Y,Z] = sphere(10)


plot3(X(:,1), Y(:,1), Z(:,1))

%% Execute rep3.m || A script which generates n random initial points for and visualise results of simulation of a 3d Hopfield network net
% Still needed here to modify the initial points

T = [1 1 1; -1 -1 1; 1 -1 -1]';
net = newhop(T);
n = 30;
[X,Y,Z] = sphere(n);

a1 = zeros(length(X(:,1))*(n+1),1);
a2 = zeros(length(a1),1);
a3 = zeros(length(a1),1);

a1(1:n+1) = X(:,1);
a2(1:n+1) = Y(:,1);
a3(1:n+1) = Z(:,1);

for i = 1:(n-1)

    a1( i*(2+n) : i*(2+n) + n) = X(:,i+1);
    a2( i*(2+n) : i*(2+n) + n) = Y(:,i+1);
    a3( i*(2+n) : i*(2+n) + n) = Z(:,i+1);
end

for i=1:length(a1)
    a = {[a1(i); a2(i) ; a3(i)]}; 
    [y,Pf,Af] = sim(net,{1 50},{},a);       % simulation of the network  for 50 timesteps
    record=[cell2mat(a) cell2mat(y)];       % formatting results
    start=cell2mat(a);                      % formatting results 
    plot3(start(1,1),start(2,1),start(3,1),'bx',record(1,:),record(2,:),record(3,:),'r');  % plot evolution
    hold on;
    plot3(record(1,50),record(2,50),record(3,50),'gO');  % plot the final point with a green circle
end

