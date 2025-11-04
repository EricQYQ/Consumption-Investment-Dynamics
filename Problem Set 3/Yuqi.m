%%% Problem set 3
%% Parameters:
clear;
beta=0.96;
gamma=3.5;
r=0.04;
rho=0.92;
sigma=0.05;

%% Discrete AR process for income when employed(Tauchen function):
mstd=3;
ny_employed=7;
ny=ny_employed+1;
[y_space,P] = Tauchen(ny_employed,0,rho,sigma,mstd);% y bar is 0
P=0.95.*P;
col_1=[0.05;0.05;0.05;0.05;0.05;0.05;0.05];
P=[col_1 P];
P=[0.25,0.75,0,0,0,0,0,0;P]; % We now have the true transitory matrix with the first state to be unemployed. 
% P{lowest income state|unemployed}=0.75
% P{unemployed|employed}=0.05
ey_space=zeros(7,1);
for i=1:7
    ey_space(i)=exp(y_space(i));
end
ey_space=[0;ey_space]; % Now have the ey_sapce with unemployment state

%% A, here the asset is the asset at the end of the period. It is w_t-c_t for each time t
%Amin=-ey_space(2)/r;
Amin=-min(ey_space)/r;
Amax=max(ey_space)/r;


%% W grid
% Since given w and y, choosing c will decide the saving. Do not need asset
% variable here.
w_min=Amin+min(ey_space);
w_max=Amax+max(ey_space);
nw=150;
w_space=linspace(w_min,w_max,nw);
w_space=w_space';

nA=nw;
A_space=linspace(Amin,Amax,nA);
A_space=A_space';

% State variable here is w only. The w' will imply the state of the world.
% (w'-y')/(1+r)=w-c
% y'=ln[w'-(w-c)(1+r)]
bc_ini= zeros(nw,ny);% Best consumption function

for i=1:nw
    for j=1:ny
        bc_ini(i,j)=w_space(i); % whatever income state, bc is w
    end
end

bc_fin=zeros(nw,ny);
dif=1;
tol=10^(-6);
max_iter=10000;
iter=0;
c_endo=zeros(nw,ny);% consumption choice is related to initial wealth at that time period and the income state of that time 
% period.
w_endo=zeros(nw,ny); %w-c=A, w=c+A, c is correlated with initial w and y. So, endogeneous w is related to initial w and y.

%% The Method of Endogenous Gridpoints
% c is a function of w now. It means given w, then c is determined. If
% asset a is fixed, then the income states y will decide future w. Then
% decide future consumption c.
tic
while dif>tol && iter<max_iter
    iter=iter+1;
    for j=1:ny
        mu_exp=zeros(nw,1);
        for k=1:ny
            w_prime=(1+r)*A_space+ey_space(k);% for each future income state, the w' is (1+r)*(w-c)+e^(y')
            %w_prime is a nw*1 vector
            c_prime=interp1(w_space,bc_ini(:,k),w_prime,'spline');% Given function of c to w, interpolate the function.Have the
            % interpolation value of c' to w'
            %c_prime is a nw*1 vector.

            c_prime=max(c_prime,tol);% consumption must be positive
            c_prime=min(c_prime,w_prime);% given debt limit is 0, consumption should be smaller than the initial wealth at that
            %period

            mu_exp=mu_exp+c_prime.^(-gamma)*P(j,k);%expected values
        end 
        c_endo=(beta*(1+r)*mu_exp).^(-1/gamma); % by Euler equation
        %c_endo is a nw*1 vector
        w_endo=c_endo+A_space;% w-c=A, w=c+A
        bc_updated=interp1(w_endo,c_endo,w_space,'spline');% The updated policy function of c to W is given by c_endo to w_endo
        % and interpolate it.
        %bc_updated is nw*1 vector.
        bc_fin(:,j)=bc_updated;
        
    end
dif=max(abs(bc_fin-bc_ini),[],"all");
bc_ini=bc_fin;

end
toc

%% Graph the consumption policy function on the w grid, for each income state:
figure
plot(w_space,bc_ini)
xlabel('Cash on hand')
ylabel('Best consumption')
title('The Plot of the Best Consumption Function to Cash on Hand for all Income Scenarios')

%% Calculate a correlogram between the simulated income and consumption series to 4 lags:
%Set seeds:
rng(2);
% Generate normal distribution with mean 0 and standard deviation sigma.
epsilon = randn(1000,1)*sigma;

% Set initial income as y_bar:
y_0=0;

% Given initial
y_sim = zeros(1000,1);
y_sim(1)=rho*y_0+epsilon(1);
for i=2:1000
    % Apply AR(1) process:
    y_sim(i)=rho*y_sim(i-1)*rho+epsilon(i);
end

y_simo=y_sim;
ey_sim=zeros(1000,1);

for i=1:1000
    ey_sim(i)=exp(y_sim(i));
end

ey_simo=ey_sim;




%Set initial wealth:
w_0=w_space(round(nw/2));
w_sim=zeros(1000,1);
w_sim(1)=w_0+exp(y_sim(1));

c_sim=zeros(1000,1);

% Find the closest grid point for ey_sim
for i=1:1000
    ey_simivec=ones(ny,1);
    ey_simivec=ey_simivec*ey_sim(i);
    [Minimum, index]=min([abs(ey_simivec-ey_space)],[],"all");
    ey_sim(i)=ey_space(index);
end     

% 

bc_sim=zeros(1000,1);
% Find the corresponding index in ey_space for ey_sim(1):
ey_sim1vec=ones(ny,1);
ey_sim1vec=ey_sim1vec*ey_sim(1);
[Minimum_1, index_1]=min([abs(ey_sim1vec-ey_space)],[],"all");
% Find the corresponding best consumption for initial wealth and y_sim(1):
bc_sim(1)=bc_ini( round(nw/2),index_1); % round function gives the nearest integer

% While loop to find the best consumption for each dynamicly updated wealth and y_sim:
i=1;
while i<=999
    i=i+1;
    % At period s, choose the y_space index of y_s(y_sim(s)). Store the
    % index as "index". Can use the cap function to recal the best a prime
    % with respecat to the (index_cap, index) pair of (a,y).
    ey_simivec=ones(ny,1);
    ey_simivec=ey_simivec*ey_sim(i);
    [Minimum, index_y]=min([abs(ey_simivec-ey_space)],[],"all");
    w_sim(i)=(w_sim(i-1)-bc_sim(i-1))*(1+r)+ey_sim(i);


    w_simivec=ones(nw,1);
    w_simivec=w_simivec*w_sim(i);
    [Minimum_w, index_w]=min([abs(w_simivec-w_space)],[],"all");%find the grid point closest to the simulated w_i

    bc_sim(i)=bc_ini(index_w,index_y); % find the best consumption

end

% Discard the first 500 observations:
ey_sim(1:500)=[];
w_sim(1:500)=[];
bc_sim(1:500)=[];

% Use the xcorr function to calculate a correlogram between the simulated 
% income and consumption series to 4 lags.

[c,lags] = xcorr(ey_sim,bc_sim,4);

% Use the corrplot function to plot your correlogram.
data=[ey_sim,bc_sim];
figure
corrplot(data,Type="Kendall",TestR="on")


%% Explain why a negative natural borrowing limit results in linear consumption functions
%with this utility function while the VFI consumption functions had curvature at the low
%end of assets:

% The VFI consumption functions had curvature at the low end of assets
% because the Inada condition: marginal utility from consuming at a very
% low level is very high.

% At the same time, a negative natural borrowing limit allow consumption
% smoothing. Consumtion will not start from 0 when initial wealth is 0.
% Consumption will start from a positive number when initial wealth is 0.
% It makes the graph smooth(linear).

%% Explain why having the unemployment state induces curvature at the low end of cash on 
% hand for this utility function

% Inducing unemployment make the lower bound of income 0. Which enforces
% consumers not to consume anything higher than initial wealth. By Inada
% condition, consumers will almost spend all they have when their initial
% wealth is very low. Then the desire decrease over wealth(marginal utility
% of consumption is decreasing). As a result, it creates the curvature at
% the low end of wealth.

















% Use the original series:
%w_simo=zeros(1000,1);
%w_simo(1)=w_0+ey_simo(1);
%bc_sim=zeros(1000,1);
%bc_sim(1)=interp2(w_space,ey_space,bc_ini,w_simo(1),ey_simo(1),"linear");



