%%% Value function iteration assignment:
%% Step 1: Coefficients and parameters

clear;

beta=0.96;
gamma=1.3;
%gamma=2.6;

r=0.04;
%r=0.08;

%keep genarilzed form of the number of a and y points for possible expansion of
%the grid
na=90;
%na=45
ny=15;

sigma=0.04;
%sigma=0.08;

rho=0.9;
mu=0;



% keep generalized fomr of maximum absolute standard deviation for possible
% expansion
abst=3;

%% Step 2: Generate state variable space for y( AR(1) process by using
%% Tauchen Function
[y_space,P] = Tauchen(ny,mu,rho,sigma,abst);
% This step will generate a state space of y and the transition matrix.
% The state space will include 5 possible points for y. The unconditional
% expectation value of y is 0. The AR(1) coefficient pho is 0.9. The
% standard deviation is 0.04. And these 5 possible points will lie within 2
% standard deviation range of the expected value.

%% Step 3: Construct grids for a
% The minimum of a is -min(y)/r

amin=-min(exp(y_space))/r;
%amin=0;

% Do not know the exact maximum of a 
amax=min(exp(y_space))/r;

% Generate the stata space for a:
a_space=linspace(amin,amax,na);
a_space=a_space';

% Set up the initial guess of value function V(a,y)
v_0=zeros(na,ny);
v_1=ones(na,ny);

% error tolerance:
errtol=10^(-9);


%% Step 4: Bellman equation:

% Consumption and utility corresponding to different a, y and a_prime:
c_space = zeros(na,ny,na);
u_space = zeros(na,ny,na);

for i = 1:na
    for j = 1:ny
        for k = 1:na
        % consumption if a is a_space(i), y is y_space(j) and a' is a_space(k)
        c_space(i,j,k) = a_space(i) + exp(y_space(j)) - a_space(k)/(1+r);        
        % utility 
          if (c_space(i,j,k)>0)
            u_space(i,j,k) = (c_space(i,j,k)^(1-gamma))/(1-gamma);
          else
            u_space(i,j,k) = -inf;
          end
        end
    end
end

% Record the time to run:
tic
%Initial difference
dif = 1;
% Initial best a_prime
cap = zeros(na, ny);
% Initial attempt value of function:
attempt = zeros(na,1);

while (dif > errtol)    
    for i = 1:na
        for j = 1:ny
            for k = 1:na
            % future states' probabilities only rely on current y.
            % Following Markov chain, prob(y_t+1=y_l|y_t=y_n)=P(n,l)
            % when the a_prime is a_space(k), the value of function is:
            exp_v=0;
              for l = 1:ny
                  exp_v=exp_v+P(j,l)*v_0(k,l);
              end
              attempt(k) = u_space(i,j,k) + beta *exp_v;
              
            end
            % Find the maximum function value and the maximum a_prime
            [Maximum,m] = max(attempt,[],"all");
            cap(i,j) = a_space(m);
            % Let the v_1(i,j) be the maximum value after the function calculation:
            v_1(i,j) = Maximum;
        end
        
        
    end       
    % Calculate distance between v_1 and v_0
    dif = max(abs(v_1 - v_0),[],"all");
    % Let the next value function's initial value be the final value of the
    % value function we just got.
    v_0 = v_1;
end
toc

% Display V(a,y)
disp(v_0);

disp(errtol);

disp(dif);

% Display a_prime(a,y)
disp(cap);

% Display corresponding consumption under a and y:
c=zeros(na,ny);
for i = 1:na
    for j = 1:ny
        % consumption if a is a_space(i), y is y_space(j) 
        c(i,j) = a_space(i) + exp(y_space(j)) - cap(i,j);        

    end
end

disp(c)

%% Step 5: Graph the converged value function in (a, V) space for all y:
figure
plot(a_space,v_0)
xlabel('Asset')
ylabel('Value of function')
title('Plot of the Value of Function to Asset for all Income')

%% Step 6: Generate a series of 1000 simulated income innovations for the 
%% given normal distribution and simulate the model for 1000 periods:
%Set seeds:
rng(2);
% Function mormrnd: generate observations following normal
% distribution with mean mu and standard deviation sigma.
epsilon = normrnd(mu, sigma, [1000, 1]);

% Give y an initial value:
y_sim = zeros(1000,1);
% Given y_0=0
y_0=0;
y_sim(1)=rho*y_0+epsilon(1);
for i=2:1000
    % Apply AR(1) process:
    y_sim(i)=y_sim(i-1)*rho+epsilon(i);

end

% Approximate y_sim to the grid points in y_space:
for i=1:1000
    y_simivec=ones(ny,1);
    y_simivec=y_simivec*y_sim(i);
    [Minimum, index]=min([abs(y_simivec-y_space)],[],"all");
    y_sim(i)=y_space(index);
end    

% Apply policy functions:
% Apply best a_prime function given a and y:
cap_sim=zeros(1000,1);
c_sim=zeros(1000,1);
% Use 15th smallest asset point as the initial asset.
% Find the corresponding index in y_space for y_sim(1):
y_sim1vec=ones(ny,1);
y_sim1vec=y_sim1vec*y_sim(1);
[Minimum_1, index_1]=min([abs(y_sim1vec-y_space)],[],"all");
% Find the corresponding best a prime and consumption for 15th smallest
% asset and y_sim(1):
cap_sim(1)=cap( round(na/2),index_1); % round function gives the nearest integer
c_sim(1)=c(round(na/2),index_1);

% While loop to find the best a prime and best consumption for each
% dynamicly updated asset and y_sim:
i=1;
while i<=999
    i=i+1;
    % At period s, choose the y_space index of y_s(y_sim(s)). Store the
    % index as "index". Can use the cap function to recal the best a prime
    % with respecat to the (index_cap, index) pair of (a,y).
    y_simivec=ones(ny,1);
    y_simivec=y_simivec*y_sim(i);
    [Minimum, index_y]=min([abs(y_simivec-y_space)],[],"all");
    % At period s, store the index of cap_sim(s-1), which is the index of
    % current a as the a index.
    % cap_sim(i-1) is the asset at period i:
    [Minimum_cap,index_a]=min(abs(a_space-cap_sim(i-1)),[],"all");
    cap_sim(i)=cap(index_a,index_y);
    c_sim(i)=c(index_a,index_y);

end

disp(cap_sim);
disp(c_sim);

% Drop first 500 observations:
y_sim(1:500)=[];
cap_sim(1:500)=[];
c_sim(1:500)=[];



%% Create a tileplot of simulated y or exp(y), a′ and c:
figure
tiledlayout(1,3);

% Tile 1
nexttile
plot(exp(y_sim))
xlabel('Periods')
ylabel('exp(y)')
title('Plot of the Income over Time')
% Tile 2
nexttile
plot(cap_sim)
xlabel('Periods')
ylabel('Best a prime')
title('Plot of the Best a prime over Time')
% Tile 3
nexttile
plot(c_sim)
xlabel('Periods')
ylabel('Best consumption')
title('Plot of the Best Consumption over Time')

%% Calculate standard deviation of c:
disp(std(c_sim));

%% What will happen to c_std when
% (a) the borrowing constraint were zero:
% If the borrowing constraint is 0, consumers will have difficulty to
% smooth consumption. The std(c) should go up.

% (b) If the relative risk aversion doubles, consumer will be more
% risk-averse and more prudent. They will smooth consumption more and do
% more precautionary saving. The std(c) should go down.

% (c)If the natural rate of interest doubled. Not sure how std(c) will
% change.

% (d) If the income volatility doubles:The expectation of the future
% permanent income will changes relatively greatly. By PIH, the current
% consumption is a function of current asset and expectation of lifetime
% income. So the consumption will change relatively greatly. As a result,

% std(c) goes up.
