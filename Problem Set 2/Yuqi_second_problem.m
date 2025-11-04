%%%% Value function iteration assignment 2 problem 1:
%%% Value function iteration:
%% Parameters:

clear;
beta=0.96;
rho=0.9;
sigma=0.15;
wbar=2.5;
phi_wage=2;
tol=10^(-9);
nbar=40/168;

alpha=0.4;
delta0=0.1;
phi2=0.2;
phi1=1/beta-(1-delta0);


% Calibration for omiga:
% Set up Lagragian function, take partial derivatives to c and n. Then have
% labor supply condition: n^(1/phi)=w/omiga
% In steady state:omiga=w_bar/n_bar^(1/phi)
omiga=wbar/nbar^(1/phi_wage);

%% Number of grid points:
nk=450;% the number of grid point for capital is 30
%nw=21;
nw=7; % nw is 7 here due to the maximum standard deviation for Tauchen function is set to 3.

%% Capital grid:
% Capital should be nonnegative
kmin=0;
kmax=30;
k_space=linspace(kmin,kmax,nk);

%% Discrete AR process for wage(Tauchen function):
mstd=3;
[w_space,P] = Tauchen(nw,wbar,rho,sigma,mstd);

%% Employment:
% By solving the firm's profit maximizing problem
% For every period t+s, the optimal employment is:
% n_t+s=(w_t+s/(1-alpha))^(-1/alpha)*u_t+s*k_t+s.

%% u and n:
% By solving the FOC to u_t+s:
% n_t+s=(phi1+phi2*u_t+s-phi2)^(1/(1-alpha))/alpha^(1/(1-alpha))*u_t+s*k*t+s

% Two equations, two unknowns. We can solve it:
u_space=zeros(nw,1);
for i=1:nw
    u_space(i)=(w_space(i)/(1-alpha))^((alpha-1)/alpha)*alpha/phi2+(phi2-phi1)/phi2;
end

n_space=zeros(nk,nw);
for i=1:nk
    for j=1:nw
        n_space(i,j)=(w_space(j)/(1-alpha))^(-1/alpha)*w_space(j)*k_space(i);
    end
end

%% Delta:
delta_space=zeros(nw,1);
for i=1:nw
    delta_space(i)=delta0+phi1*(u_space(i)-1)+phi2/2*(u_space(i)-1)^2;
end

%% Value function iteration:
v_ini=zeros(nk,nw);
v_fin=ones(nk,nw);
%v_exp=zeros(nk,nw);%v_exp(i,j) is the expect value when k'=k_space(i) and current wage state is j. v_exp=v_ini*P'


dif = 1;

bkp=zeros(nk,nw);

attempt = zeros(nk,1);

%% Build current prifit function:
prof=zeros(nk,nw,nk);
for i=1:nk
    for j=1:nw
        for l=1:nk
            prof(i,j,l)=(u_space(j)*k_space(i))^alpha+(1-delta_space(j))*k_space(i)-k_space(l);
        end
    end
end




tic % It will take around 15 seconds to run it.

while (dif > tol)    
    for i = 1:nk
        for j = 1:nw 
            for l = 1:nk
                exp_v=0;%for expected value of value function
                for m=1:nw
                    exp_v=exp_v+P(j,m)*v_ini(l,m);
                end
                attempt(l) = prof(i,j,l) + beta *exp_v;
            end
              

            % Choosing the next period's asset to maximize expected utility
            % given asset and wage.
            [Maximum,m] = max(attempt);
            bkp(i,j) = k_space(m);
            % Let the v_1(i,j) be the maximum value after the function calculation:
            v_fin(i,j) = Maximum; % The final value of value function is the maximum by choosing the next period's asset and working hours.
        end
        
        
    end       

    dif = max(abs(v_fin - v_ini),[],"all");
    v_ini = v_fin;
end
toc

%% Display value function, best k prime, and best u, best n, and best inv:

disp(v_fin);
disp(bkp);
disp(u_space);
disp(n_space);
invp=zeros(nk,nw);
for i=1:nk
    for j=1:nw
        invp(i,j)=bkp(i,j)-(1-delta_space(j))*k_space(i);
    end
end
disp(invp);



figure
plot(k_space,v_ini)
xlabel('Capital')
ylabel('Value of function')
title('The Plot of the Value of Function to Capital for all Wage Scenarios')

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
% initial values:
bkp_sim=zeros(1000,1);
n_sim=zeros(1000,1);
invp_sim=zeros(1000,1);
u_sim=zeros(1000,1);

% For u:
for i =1:1000
  u_sim(i)=(w_sim(i)/(1-alpha))^((alpha-1)/alpha)*alpha/phi2+(phi2-phi1)/phi2;
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

% Use the nk/2 th smallest capital point as the initial capital.
% Find the corresponding index in w_space for w_sim(1):
w_sim1vec=ones(nw,1);
w_sim1vec=w_sim1vec*w_sim(1);
[Minimum_1, index_1]=min([abs(w_sim1vec-w_space)],[],"all");
% Find the corresponding best a prime and consumption for the nk/2 th smallest
% asset and w_sim(1):
bkp_sim(1)=bkp( round(nk/2),index_1); % round function gives the nearest integer
n_sim(1)=(w_sim(1)/(1-alpha))^(-1/alpha)*w_sim(1)*k_space(round(nk/2));
invp_sim(1)=invp(round(nk/2),index_1);

% While loop to find the best a prime and best consumption for each
% dynamicly updated asset and w_sim:
i=1;
while i<=999
    i=i+1;
    % At period s, choose the w_space index of w_s(w_sim(s)). Store the
    % index as "index". Can use the cap function to recal the best k prime
    % with respecat to the (index_bkp, index) pair of (k,w).
    w_simivec=ones(nw,1);
    w_simivec=w_simivec*w_sim(i);
    [Minimum, index_w]=min([abs(w_simivec-w_space)],[],"all");
    % At period s, store the index of bkp_sim(s-1), which is the index of
    % current k as the k index.
    % bkp_sim(i-1) is the capital at period i:
    [Minimum_bkp,index_k]=min(abs(k_space-bkp_sim(i-1)),[],"all");
    bkp_sim(i)=bkp(index_k,index_w);
    n_sim(i)=(w_sim(i)/(1-alpha))^(-1/alpha)*w_sim(i)*k_space(index_k);
    invp_sim(i)=invp(index_k,index_w);

end

disp(bkp_sim);
disp(u_sim);
disp(n_sim);
disp(invp_sim);

% Drop first 500 observations:
w_simori(1:500)=[];
bkp_sim(1:500)=[];
u_sim(1:500)=[];
n_sim(1:500)=[];
invp_sim(1:500)=[];



%% Create a tileplot of simulated w, a′, n and c:
figure
tiledlayout(1,5);

% Tile 1
nexttile
plot(w_simori)
xlabel('Periods')
ylabel('Wage simulations')
title('Plot of the Wage over Time')

% Tile 2
nexttile
plot(bkp_sim)
xlabel('Periods')
ylabel('Best capital prime')
title('Plot of the Best Capital Prime over Time')


% Tile 3
nexttile
plot(u_sim)
xlabel('Periods')
ylabel('Best u')
title('Plot of the Best u over Time')


% Tile 4
nexttile
plot(n_sim)
xlabel('Periods')
ylabel('Best employment')
title('Plot of the Best Employment over Time')

% Tile 5
nexttile
plot(invp_sim)
xlabel('Periods')
ylabel('Best investment')
title('Plot of the Best Investment over Time')% The plots look very steep 
% because having approximated the continuous variables to the closest grid
% points.

%% Calculate standard deviation of n:
disp(std(n_sim));
disp(std(u_sim));
disp(std(invp_sim));

%% %% Explain what you would qualitatively expect would occur to the standard deviation of each in the following cases:
% u=(w/(1-alpha))^((alpha-1)/alpha)*alpha/phi2+(phi2-phi1)/phi2;
% n=(w/(1-alpha))^(-1/alpha)*w*k
% inv=k'-(1-delta0+phi1*(u-1)+phi2/2*(u-1)^2)k

%% (a) If phi2 doubles.
% u=(w/(1-alpha))^((alpha-1)/alpha)*alpha/phi2+1-phi1/phi2. It still
% depends on w to determine the effect on u. So, do not know on u.

% n=(w/(1-alpha))^(-1/alpha)*w*k. It depends on how u will change. Since
% the effect on u is not certain. The effect on n is also not certain here.

% inv=k'-(1-delta0+phi1*(u-1)+phi2/2*(u-1)^2)k. It will be jointly decided
% by phi2 and u. So, the effect on inv is also not certain here.

%% (b) The real interest rate doubles, say phi1 is higher:

% u=(w/(1-alpha))^((alpha-1)/alpha)*alpha/phi2+(phi2-phi1)/phi2; The
% expectation of u goes down. But the uncertain part of u is not affected
% by phi1. std(u) will not change.

% n=(w/(1-alpha))^(-1/alpha)*w*k; The k pattern may change. The overall
% effect on std(n) is not certain.

% inv=k'-(1-delta0+phi1*(u-1)+phi2/2*(u-1)^2)k; phi1 will affect inv here.
% But k pattern may change. As a result, the overall effect on std(inv) is
% not certain.

