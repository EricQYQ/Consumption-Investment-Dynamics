%%%% Value function iteration assignment 2 problem 1:
%%% Value function iteration:
%% Parameters:

clear;
beta=0.96;
gamma=1;
r=0.04;
rho=0.9;
sigma=0.15;
wbar=2.5;
phi=2;
tol=10^(-9);
nbar=40/168;

% Calibration for omiga:
% Set up Lagragian function, take partial derivatives to c and n. Then have
% labor supply condition: n^(1/phi)=w/omiga
% In steady state:omiga=w_bar/n_bar^(1/phi)
omiga=wbar/nbar^(1/phi);

%% Number of grid points:
na=90;
%nw=21;
nw=7; % nw is 7 here due to the maximum standard deviation for Tauchen function is set to 3.

%% Discrete AR process(Tauchen function):
mstd=3;
[w_space,P] = Tauchen(nw,wbar,rho,sigma,mstd);

%% Working hour from labor supply condition.
% By setting Lagrangian, deriving FOCs, the labor supply condition is:
% n_t+s^(1/phi)=w_t+s/omiga
n_space=zeros(nw,1);
for i=1:nw
    n_space(i)=w_space(i)^phi/omiga^phi;
end

%% Asset grid points:
% Income per period is labor income wn
% The natural debt limit was -income_lowbar/r
% So, the natural debt limit here is -w_lowbar*n_upperbar/r.
% w_lowbar*n_upperbar is the income working the maximum time every day with lowest
% wage.
nmin=min(n_space);
nmax=max(n_space);
wmin=min(w_space);
wmax=max(w_space);


amin = -(wmin*nmax)/r;

amax = wmin*nmax/r;
% Generate asset grid points:
a_space=linspace(amin,amax,na);
a_space=a_space';

%% Consumption and utility space:
% Knowing a,w,n,a', then c is determined:

c_space = zeros(na,nw,na);
u_space = zeros(na,nw,na);

for i = 1:na
    for j = 1:nw
        for k = 1:na
            % consumption if a is a_space(i), y is y_space(j) and a' is a_space(k)
            c_space(i,j,k) = a_space(i) + w_space(j)*n_space(j)- a_space(k)/(1+r);        
            % utility 
            if c_space(i,j,k)-omiga*(n_space(j)^(1+1/phi))/(1+1/phi)>0
              u_space(i,j,k) = log(c_space(i,j,k)-omiga*(n_space(j)^(1+1/phi))/(1+1/phi));
            else
              u_space(i,j,k) = -inf;
            end
        end
    end
end

%% Value function iteration:
v_ini=zeros(na,nw);
v_fin=ones(na,nw);

dif = 1;

bap=zeros(na,nw);

attempt = zeros(na,1);


tic

while (dif > tol)    
    for i = 1:na
        for j = 1:nw 
            for k = 1:na
                exp_v=0;%for expected value of value function
                for l=1:nw
                   exp_v=exp_v+P(j,l)*v_ini(k,l);
                end
                attempt(k) = u_space(i,j,k) + beta *exp_v;
            end
              

            % Choosing the next period's asset to maximize expected utility
            % given asset and wage.
            [Maximum,m] = max(attempt);
            bap(i,j) = a_space(m);
            % Let the v_1(i,j) be the maximum value after the function calculation:
            v_fin(i,j) = Maximum; % The final value of value function is the maximum by choosing the next period's asset and working hours.
        end
        
        
    end       

    dif = max(abs(v_fin - v_ini),[],"all");
    v_ini = v_fin;
end
toc


%% Display value function, best a prime, and best c:
c=zeros(na,nw);
for i = 1:na
    for j = 1:nw
        c(i,j) = a_space(i) + w_space(j)*n_space(j)- bap(i,j)/(1+r); % Use budget constraint here.
    end
end
disp(v_fin);
disp(bap);
disp(c);


figure
plot(a_space,v_ini)
xlabel('Asset')
ylabel('Value of function')
title('The Plot of the Value of Function to Asset for all Wage Scenarios')

%%% Simulation:
%% Simulation:
%Set seeds:
rng(2);
% Generate normal distribution with mean 0 and standard deviation sigma.
epsilon = randn(1000,1)*sigma;

% Wage:
w_sim = zeros(1000,1);
w_0=wbar; % Initial value is steady state wage.
w_sim(1)=(1-rho)*wbar+rho*w_0+epsilon(1);
for i=2:1000
    % Apply AR(1) process:
    w_sim(i)=(1-rho)*wbar+rho*w_sim(i-1)*rho+epsilon(i);

end
w_simori=w_sim;

% n:

n_sim=zeros(1000:1);

for i=1:1000
  n_sim(i)=w_sim(i)^phi/omiga^phi;
end



% Approximate w_sim to the grid points in w_space:
for i=1:1000
    w_simivec=ones(nw,1);
    w_simivec=w_simivec*w_sim(i);
    [Minimum, index]=min([abs(w_simivec-w_space)],[],"all");
    w_sim(i)=w_space(index);
end    

% Apply policy functions:
% Apply best a_prime function given a and w:
bap_sim=zeros(1000,1);
c_sim=zeros(1000,1);
% Use the 3*na/4 th asset point as the initial asset.
% Find the corresponding index in w_space for w_sim(1):
w_sim1vec=ones(nw,1);
w_sim1vec=w_sim1vec*w_sim(1);
[Minimum_1, index_1]=min([abs(w_sim1vec-w_space)],[],"all");
% Find the corresponding best a prime and consumption for the 3*na/4 th smallest
% asset and w_sim(1):
bap_sim(1)=bap( round(3*na/4),index_1); % round function gives the nearest integer
c_sim(1)=c(round(3*na/4),index_1);

% While loop to find the best a prime and best consumption for each
% dynamicly updated asset and w_sim:
i=1;
while i<=999
    i=i+1;
    % At period s, choose the w_space index of w_s(w_sim(s)). Store the
    % index as "index". Can use the cap function to recal the best a prime
    % with respecat to the (index_bap, index) pair of (a,w).
    w_simivec=ones(nw,1);
    w_simivec=w_simivec*w_sim(i);
    [Minimum, index_w]=min([abs(w_simivec-w_space)],[],"all");
    % At period s, store the index of bap_sim(s-1), which is the index of
    % current a as the a index.
    % bap_sim(i-1) is the asset at period i:
    [Minimum_bap,index_a]=min(abs(a_space-bap_sim(i-1)),[],"all");
    bap_sim(i)=bap(index_a,index_w);
    c_sim(i)=c(index_a,index_w);

end

disp(bap_sim);
disp(n_sim);
disp(c_sim);

% Drop first 500 observations:
w_simori(1:500)=[];
bap_sim(1:500)=[];
n_sim(1:500)=[];
c_sim(1:500)=[];



%% Create a tileplot of simulated w, a′, n and c:
figure
tiledlayout(1,4);

% Tile 1
nexttile
plot(w_simori)
xlabel('Periods')
ylabel('Wage simulations')
title('Plot of the Wage over Time')

% Tile 2
nexttile
plot(bap_sim)
xlabel('Periods')
ylabel('Best a prime')
title('Plot of the Best a prime over Time')

% Tile 3
nexttile
plot(n_sim)
xlabel('Periods')
ylabel('Best working hours')
title('Plot of the Best Working Hours over Time')

% Tile 4
nexttile
plot(c_sim)
xlabel('Periods')
ylabel('Best consumption')
title('Plot of the Best Consumption over Time')% The plots look very steep 
% because having approximated the continuous variables to the closest grid
% points.

%% Calculate standard deviation of n:
disp(std(n_sim));


%% Explain what you would qualitativel expect would occur to the standard deviation of n in the following cases:
% n=w^phi/omiga^phi;
% (a) If the borrowing constraint is 0. It will not affect n directly. The
% std(n) does not change.

% (b) If the relative risk aversion parameter doubles. It also will not
% affect n directly. The std(n) does not change.

% (c) If the Frisch labor supply elasticity doubles. Phi doubles, 
% n=(w/omiga)^phi. Std(n) goes up.

% (d) If real wage volatility doubles. n=w^phi/omiga^phi, std(n) goes up.

