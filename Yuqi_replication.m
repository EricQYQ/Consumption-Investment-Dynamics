%% %% %% Replication Assignment:
%% %% Start of question 1 and 2:
%% Parameters:

clear;

beta=0.9;
lambda=0.75;
F=0.2;
delta=0.1;

%% Two state Markov process:
P=[0.9,0.1;0.1,0.9];
A_space=[1.25,0.75]';

%% z space:
z_space=[0,1]';

%% Epsilon space:
epsilon_space=linspace(0.4,1.6,20);
epsilon_space=epsilon_space';

%% k space
nk=20;
k_range=ones(1,nk)';
k_range=k_range*(1-delta);
k_range(1)=1;
k_space=cumprod(k_range);

%% Value function preparation
v_0=zeros(nk,2,20);
v_1=ones(nk,2,20);

% Tolerance:
mu=10^(-9);


% Initial V^R, V^NR,z:
vR=zeros(nk,2,20);
vNR=zeros(nk,2,20);
bestz=zeros(nk,2,20);



%% Value function:
tic
%Initial difference
dif = 1;
% Initial attempt value of function:
count=0;

while dif > mu & count<1000
    count=count+1;
    for i = 1:nk
        for j = 1:2
            for s = 1:20
            % future states' probabilities only rely on current y.
            % Following Markov chain, prob(y_t+1=y_l|y_t=y_n)=P(n,l)
            % when the a_prime is a_space(k), the value of function is:
            exp_vR=0;
            exp_vNR=0;
              for l = 1:2
                  for x = 1:20
                    exp_vR=exp_vR+0.05*P(j,l)*v_0(1,l,x);
                    if i<nk
                    exp_vNR=exp_vNR+0.05*P(j,l)*v_0(i+1,l,x);
                    else 
                    exp_vNR=exp_vNR+0.05*P(j,l)*v_0(i,l,x);
                    end
                  end
              end
              vR(i,j,s) = A_space(j)*epsilon_space(s)*lambda*k_space(i)-F + beta *exp_vR;
              vNR(i,j,s) = A_space(j)*epsilon_space(s)*k_space(i) + beta *exp_vNR;
              if vR(i,j,s)>=vNR(i,j,s)
                  bestz(i,j,s)=1;
              else
                  bestz(i,j,s)=0;
              end
              % Let the v_1(i,j,s) be the maximum value after the function calculation:
              v_1(i,j,s) = max(vR(i,j,s),vNR(i,j,s));
            end

        end
        
        
    end       
    % Calculate distance between v_1 and v_0
    dif = max(abs(v_1 - v_0),[],"all");
    % Let the next value function's initial value be the final value of the
    % value function we just got.
    v_0 = v_1;
end
toc

%% %% Start of question 3:
%% Always replace(bestz=1) since the 5th largest k, 5th largest k is (1-delta)^4. So, let nk=6 and try:
k_space2=k_space(1:6);
nk2=6;
% Initial V^R, V^NR,z,v_0,v_1:
vR2=zeros(nk2,2,20);
vNR2=zeros(nk2,2,20);
bestz2=zeros(nk2,2,20);
v_02=zeros(nk2,2,20);
v_12=ones(nk2,2,20);

%% Value function:
tic
%Initial difference
dif2 = 1;
% Initial attempt value of function:
count2=0;

while dif2 > mu & count2<1000
    count2=count2+1;
    for i = 1:nk2
        for j = 1:2
            for s = 1:20
            % future states' probabilities only rely on current y.
            % Following Markov chain, prob(y_t+1=y_l|y_t=y_n)=P(n,l)
            % when the a_prime is a_space(k), the value of function is:
            exp_vR2=0;
            exp_vNR2=0;
              for l = 1:2
                  for x = 1:20
                    exp_vR2=exp_vR2+0.05*P(j,l)*v_0(1,l,x);
                    if i<nk2
                    exp_vNR2=exp_vNR2+0.05*P(j,l)*v_0(i+1,l,x);
                    else 
                    exp_vNR2=exp_vNR2+0.05*P(j,l)*v_0(i,l,x);
                    end
                  end
              end
              vR2(i,j,s) = A_space(j)*epsilon_space(s)*lambda*k_space(i)-F + beta *exp_vR2;
              vNR2(i,j,s) = A_space(j)*epsilon_space(s)*k_space(i) + beta *exp_vNR2;
              if vR2(i,j,s)>=vNR2(i,j,s)
                  bestz2(i,j,s)=1;
              else
                  bestz2(i,j,s)=0;
              end
              % Let the v_1(i,j,s) be the maximum value after the function calculation:
              v_12(i,j,s) = max(vR2(i,j,s),vNR2(i,j,s));
            end

        end
        
        
    end       
    % Calculate distance between v_1 and v_0
    dif2 = max(abs(v_12 - v_02),[],"all");
    % Let the next value function's initial value be the final value of the
    % value function we just got.
    v_02 = v_12;
end
toc

%% Plot the policy function:
figure;
spy(bestz2);

% Comments:
% (1) It is more likely for a firm to replace when the idiosyncratic shock is
% low. 
% (2) And it is more likely for a firm to replace when the number of time periods
% since last replacement getting larger.
% (3) The high state has a higher replacement rate than the low state.

%% Plot the hazard function:
High_state = bestz2(:,1,:);
High_state= reshape(High_state,nk2,20);
Low_state = bestz2(:,2,:);
Low_state= reshape(Low_state,nk2,20);

% If the initial state is high, prob(z_0=0)=1.
% Prob(A_1=H)=0.9,Prob(A_1=L)=0.1.
% Prob(z_1=0|z_0=0)=0.9*0.8+0.1*19/20=0.815
% Prob(A_2=H)=0.82,Prob(A_2=L)=0.18. 
% Prob(z_2=0|z_1=0,z_0=0)=0.45*0.82+0.55*0.18=0.468.
% Prob(z_0=0,z_1=0)=1*0.815=0.815
% Prob(z_0=0,z_1=0,z_2=0)=Prob(z_2=0|z_0=0,z_1=0)*Prob(z_0=0,z_1=0)=0.815*0.468=0.3814
% Prob(A_3=H)=0.82*0.9+0.18*0.1=0.756; Prob(A_3=L)=1-0.756=0.244
% Prob(z_3=0|z_2,z_1,z_0=0)=0.756*3/20+0.244*4/20=0.1622
% Prob(z_3=0,z_2,z_1,z_0=0)=0.1622*0.3814=0.0619
f_H=[0;1-0.815;1-0.815*0.468;1-0.1622*0.3814;1;1;1;1]';
x_H=[1;2;3;4;5;6;7;8]';

% If the initial state is low, prob(z_0=0)=1.
% Prob(A_1=L)=0.9,Prob(A_1=H)=0.1.
% Prob(z_1=0|z_0=0)=0.1*0.8+0.9*19/20=0.935
% Prob(A_2=L)=0.82,Prob(A_2=H)=0.18. 
% Prob(z_2=0|z_1=0,z_0=0)=0.45*0.18+0.55*0.82=0.532.
% Prob(z_0=0,z_1=0)=1*0.935=0.935
% Prob(z_0=0,z_1=0,z_2=0)=Prob(z_2=0|z_0=0,z_1=0)*Prob(z_0=0,z_1=0)=0.935*0.532=0.4974
% Prob(A_3=L)=0.82*0.9+0.18*0.1=0.756; Prob(A_3=H)=1-0.756=0.244
% Prob(z_3=0|z_2,z_1,z_0=0)=0.244*3/20+0.756*4/20=0.1878
% Prob(z_3=0,z_2,z_1,z_0=0)=0.1878*0.4974=0.0934
f_L=[0;1-0.935;1-0.935*0.532;1-0.1878*0.4974;1;1;1;1]';

figure();
plot(x_H,f_H,'-r','LineWidth',1.5);
ylim([0, 1]);
hold on;
plot(x_H,f_L,'-b','LineWidth',1.5);
legend('High state','Low state');
xlabel('t');
ylabel('F(x)');
hold off;


% Comments:
% (1) Both in high state and low state, a firm will surely replace after 4
% peroids without replacing.

% (2) The low state has a slightly higher replacement rate than the high
% state.

%% %%Starting question 6:
%% Simulate a time series for the evolution of one firm
rng("default");
shocks=unifrnd(0.4,1.6,[40 1]);

% Create Markov chain following P:
mc= dtmc(P); % Function dtmc give a Markov chain following transition matrix P

[states_series] = simulate(mc, 39);

% Start from K_0=1:
% bestz_firstperiod=bestz(1,1,epsilon_firstperiod)
bestz_series=zeros(40,1);
capitalstock_series=ones(40,1);
output_series=zeros(40,1);
index=zeros(40,1);
shocks_topoints=zeros(40,1);

% Find the closest point for epsilons:
for i=1:40
    epsilon_simivec=ones(20,1);
    epsilon_simivec=epsilon_simivec*shocks(i);
    [Minimum, index_epsilon]=min([abs(epsilon_simivec-epsilon_space)],[],"all");
    index(i)=index_epsilon;
    shocks_topoints(i)=epsilon_space(index_epsilon);
end   


bestz_series(1)=bestz2(1,states_series(1),index(1));
if bestz_series(1)==1
    capitalstock_series(1)=1;
    output_series(1)=A_space(states_series(1))*shocks(1)*lambda*1-F;
else
    capitalstock_series(1)=1*(1-delta);
    output_series(1)=A_space(states_series(1))*shocks(1)*1;
end



i=1;

tic
while i<40
    i=i+1;
    % find the capital at the end of last period
    capital_simivec=ones(nk2,1);
    capital_simivec=capital_simivec*capitalstock_series(i-1);
    [Minimum, index_k]=min([abs(capital_simivec-k_space2)],[],"all");
    bestz_series(i)=bestz2(index_k,states_series(i),index(i));
    if bestz_series(i)==1
       capitalstock_series(i)=1;
       output_series(i)=A_space(states_series(i))*shocks(i)*lambda*k_space2(index_k)-F;
    else
       capitalstock_series(i)=capitalstock_series(i-1)*(1-delta);
       output_series(i)=A_space(states_series(i))*shocks(i)*k_space2(index_k);
    end

end
toc

figure();
plot(capitalstock_series);
hold on;
plot(output_series);
legend('capital stock series','output series');
xlabel('t');
hold off;

% The capital stock series look like following a white noise process. It is
% stationary and range from 0.7 to 1.

% The output series, however, looks like expectation stationary but not
% covariance stationary. And the variance is quite larger than the capital
% stock series.

%% %% Starting question 7:
%% The policy function may be different since only one aggregate state is available:
nk3=50;
k_range3=ones(1,nk3)';
k_range3=k_range3*(1-delta);
k_range3(1)=1;
k_space3=cumprod(k_range3);
A_H=1.25;
A_L=0.75;

%% Value function preparation
v_03=zeros(nk3,20);
v_13=ones(nk3,20);


% Initial V^R, V^NR,z:
vR3=zeros(nk3,20);
vNR3=zeros(nk3,20);
bestz3=zeros(nk3,20);



%% Value function:
tic
%Initial difference
dif = 1;
% Initial attempt value of function:
count=0;

while dif > mu & count<1000
    count=count+1;
    for i = 1:nk3
            for s = 1:20
            % future states' probabilities only rely on current y.
            % Following Markov chain, prob(y_t+1=y_l|y_t=y_n)=P(n,l)
            % when the a_prime is a_space(k), the value of function is:
            exp_vR3=0;
            exp_vNR3=0;
                  for x = 1:20
                    exp_vR3=exp_vR3+0.05*v_03(1,x);
                    if i<nk
                    exp_vNR3=exp_vNR3+0.05*v_03(i+1,x);
                    else 
                    exp_vNR3=exp_vNR3+0.05*v_03(i,x);
                    end
                  end
              vR3(i,s) = A_H*epsilon_space(s)*lambda*k_space3(i)-F + beta *exp_vR3;
              vNR3(i,s) = A_H*epsilon_space(s)*k_space3(i) + beta *exp_vNR3;
              if vR3(i,s)>=vNR3(i,s)
                  bestz3(i,s)=1;
              else
                  bestz3(i,s)=0;
              end
              % Let the v_1(i,j,s) be the maximum value after the function calculation:
              v_13(i,s) = max(vR3(i,s),vNR3(i,s));
            end

        
        
    end       
    % Calculate distance between v_1 and v_0
    dif = max(abs(v_13 - v_03),[],"all");
    % Let the next value function's initial value be the final value of the
    % value function we just got.
    v_03 = v_13;
end
toc

%% Assume there is nf=1000 firms in the economy, and the economy last for np=40 periods
np=40;
nf=1000;
disaggshocks=unifrnd(0.4,1.6,[np nf]);
%% As a result, should have shocks table of 40*1000
bestz_seriesfixed=zeros(np,nf);
capitalstock_seriesfixed=zeros(np,nf);
output_seriesfixed=zeros(np,nf);

% For these 1000 firms we have:
% Find the closest point for epsilons:
disaggindex_fixed=ones(np,nf);
disaggshocks_topointsfixed=zeros(np,nf);
for j=1:nf
for i=1:np
    epsilon_simivecfixed=ones(20,1);
    epsilon_simivecfixed=epsilon_simivecfixed*disaggshocks(i,j);
    [Minimum, index_epsilonfixed]=min([abs(epsilon_simivecfixed-epsilon_space)],[],"all");
    disaggindex_fixed(i,j)=index_epsilonfixed;
    disaggshocks_topointsfixed(i,j)=epsilon_space(index_epsilonfixed);
end   
end

%% For each firm j:

for j=1:nf
bestz_seriesfixed(1,j)=bestz3(1,disaggindex_fixed(1,j));
if bestz_seriesfixed(1,j)==1
    capitalstock_seriesfixed(1,j)=1;
    output_seriesfixed(1,j)=A_H*disaggshocks(1,j)*lambda*1-F;
else
    capitalstock_seriesfixed(1,j)=1*(1-delta);
    output_seriesfixed(1,j)=A_H*disaggshocks(1,j)*1;
end
end




tic
for j=1:nf
i=1;
while i<np
    i=i+1;
    % find the capital at the end of last period
    capital_simivecfixed=ones(nk3,1);
    capital_simivecfixed=capital_simivecfixed*capitalstock_seriesfixed(i-1,j);
    [Minimum, index_kfixed]=min([abs(capital_simivecfixed-k_space3)],[],"all");
    bestz_seriesfixed(i,j)=bestz3(index_kfixed,disaggindex_fixed(i,j));
    if bestz_seriesfixed(i,j)==1
       capitalstock_seriesfixed(i,j)=1;
       output_seriesfixed(i,j)=A_H*disaggshocks(i,j)*lambda*k_space3(index_kfixed)-F;
    else
       capitalstock_seriesfixed(i,j)=capitalstock_seriesfixed(i-1,j)*(1-delta);
       output_seriesfixed(i,j)=A_H*disaggshocks(i,j)*k_space3(index_kfixed);
    end

end
end
toc

replacement_rate=mean(bestz_seriesfixed,2);
figure();
plot(replacement_rate,'-b','LineWidth',1.3);
xlabel('Period');
ylabel('Investment Rate');
title('Without aggregate shocks');

%% %% Starting question 8:
% The policy function should follows the bestz2
% Assume there is nf=1000 firms in the economy, and the economy last for np_unfixed=160 periods


%% Assume there is nf=1000 firms in the economy, and the economy last for np=40 periods
%% As a result, should have shocks table of 40*1000
np_unfixed=160;
nf=1000;
disaggshocks_unfixed=unifrnd(0.4,1.6,[np_unfixed nf]);


% For these 1000 firms we have:
% Find the closest point for epsilons:
disaggindex_unfixed=ones(np_unfixed,nf);
disaggshocks_topointsunfixed=zeros(np_unfixed,nf);
for j=1:nf
for i=1:np_unfixed
    epsilon_simivecunfixed=ones(20,1);
    epsilon_simivecunfixed=epsilon_simivecunfixed*disaggshocks_unfixed(i,j);
    [Minimum, index_epsilonunfixed]=min([abs(epsilon_simivecunfixed-epsilon_space)],[],"all");
    disaggindex_unfixed(i,j)=index_epsilonunfixed;
    disaggshocks_topointsunfixed(i,j)=epsilon_space(index_epsilonunfixed);
end   
end


%% Aggregate state simulations:
[aggstates_series] = simulate(mc, 159);


%% For each firm j:

bestz_seriesunfixed=zeros(np_unfixed,nf);
capitalstock_seriesunfixed=zeros(np_unfixed,nf);
output_seriesunfixed=zeros(np_unfixed,nf);

% First period:
for j=1:nf
bestz_seriesunfixed(1,j)=bestz2(1,aggstates_series(1),disaggindex_unfixed(1,j));
if bestz_seriesfixed(1,j)==1
    capitalstock_seriesunfixed(1,j)=1;
    output_seriesfixed(1,j)=A_space(aggstates_series(1))*disaggshocks_unfixed(1,j)*lambda*1-F;
else
    capitalstock_seriesunfixed(1,j)=1*(1-delta);
    output_seriesfixed(1,j)=A_space(aggstates_series(1))*disaggshocks_unfixed(1,j)*1;
end
end



tic
for j=1:nf
i=1;
while i<np_unfixed
    i=i+1;
    % find the capital at the end of last period
    capital_simivecunfixed=ones(nk2,1);
    capital_simivecunfixed=capital_simivecunfixed*capitalstock_seriesunfixed(i-1,j);
    [Minimum, index_kunfixed]=min([abs(capital_simivecunfixed-k_space2)],[],"all");
    bestz_seriesunfixed(i,j)=bestz2(index_kunfixed,aggstates_series(i),disaggindex_unfixed(i,j));
    if bestz_seriesunfixed(i,j)==1
       capitalstock_seriesunfixed(i,j)=1;
       output_seriesunfixed(i,j)=A_space(aggstates_series(i))*disaggshocks_unfixed(i,j)*lambda*k_space2(index_kunfixed)-F;
    else
       capitalstock_seriesunfixed(i,j)=capitalstock_seriesunfixed(i-1,j)*(1-delta);
       output_seriesunfixed(i,j)=A_space(aggstates_series(i))*disaggshocks_unfixed(i,j)*k_space2(index_kunfixed);
    end

end
end
toc

replacement_rateunfixed=mean(bestz_seriesunfixed,2);
figure();
plot(replacement_rateunfixed);
ylim([0.2, 0.4]);
xlabel('Period');
ylabel('Investment Rate');
title('With aggregate shocks');


%% Comments:
% (1) In the no-aggregated shocks case, the aggregated investment rate
% fluctuates less and less intensive, and finally converge to around 0.3. 
% After around 15 periods, the investment rate of the whole economy enter
% the steady state of around 0.3.

% (2) However, when the aggregated risk is present, the fluctuation in the 
% aggregated investment rate is always present. And the aggregated 
% investment rate fluctuates around 0.3. It also seems that after around 15
% periods, the aggregated investment rate is covariance stationary.

% (3) The model suggests that the long run fluctuation of the aggregated 
% investment rate may only due to the aggregated economy shocks instead of 
% the aggregation of the firms' idiosyncratic shocks.
