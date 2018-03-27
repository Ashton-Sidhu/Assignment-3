clear all;
clc
format long;

Nout  = 100000; % number of out-of-sample scenarios
Nin   = 5000;   % number of in-sample scenarios
Ns    = 5;      % number of idiosyncratic scenarios for each systemic

C = 8;          % number of credit states

% Filename to save out-of-sample scenarios
filename_save_out  = 'scen_out';

% Read and parse instrument data
instr_data = dlmread('instrum_data.csv', ',');
instr_id   = instr_data(:,1);           % ID
driver     = instr_data(:,2);           % credit driver
beta       = instr_data(:,3);           % beta (sensitivity to credit driver)
recov_rate = instr_data(:,4);           % expected recovery rate
value      = instr_data(:,5);           % value
prob       = instr_data(:,6:6+C-1);     % credit-state migration probabilities (default to A)
exposure   = instr_data(:,6+C:6+2*C-1); % credit-state migration exposures (default to A)
retn       = instr_data(:,6+2*C);       % market returns

K = size(instr_data, 1); % number of  counterparties

% Read matrix of correlations for credit drivers
rho = dlmread('credit_driver_corr.csv', '\t');
sqrt_rho = (chol(rho))'; % Cholesky decomp of rho (for generating correlated Normal random numbers)
% X = randn(50,50) * sqrt_rho;
% covar = cov(X);
%covar = rho;

disp('======= Credit Risk Model with Credit-State Migrations =======')
disp('============== Monte Carlo Scenario Generation ===============')
disp(' ')
disp(' ')
disp([' Number of out-of-sample Monte Carlo scenarios = ' int2str(Nout)])
disp([' Number of in-sample Monte Carlo scenarios = ' int2str(Nin)])
disp([' Number of counterparties = ' int2str(K)])
disp(' ')

% Find credit-state for each counterparty
% 8 = AAA, 7 = AA, 6 = A, 5 = BBB, 4 = BB, 3 = B, 2 = CCC, 1 = default
[Ltemp, CS] = max(prob, [], 2);
clear Ltemp

% Account for default recoveries
exposure(:, 1) = (1-recov_rate) .* exposure(:, 1);

% Compute credit-state boundaries
CS_Bdry = norminv( cumsum(prob(:,1:C-1), 2) );

% -------- Insert your code here -------- %
if(~exist('scenarios_out.mat','file'))
    
    % -------- Insert your code here -------- %
    
    parfor s = 1:Nout        
        normcorr = sqrt_rho *randn(50,1);
        w = [];
        for i = 1:K           
            norm = normrnd(0,1);
            stddev = sqrt(1-(beta(i)^2));
            credit = (beta(i) * normcorr(driver(i))) + (stddev * norm);            
            for c = 1:size(CS_Bdry,2)
                credBound = CS_Bdry(i, :);
                if(credit < credBound(c))
                    lossIndex = c;
                    break                
                else
                    lossIndex = 7;
                end            
                
            end
           
            loss = exposure(i,lossIndex);
            w = [w; loss];           
        end
       Losses_out(s, :) = w;
    end

    % Calculated out-of-sample losses (100000 x 100)
    % Losses_out
   
    save('scenarios_out', 'Losses_out')
else
    load('scenarios_out', 'Losses_out')
end

% Normal approximation computed from out-of-sample scenarios
mu_l = mean(Losses_out)';
var_l = cov(Losses_out);

% Compute portfolio weights
portf_v = sum(value);     % portfolio value
w0{1} = value / portf_v;  % asset weights (portfolio 1)
w0{2} = ones(K, 1) / K;   % asset weights (portfolio 2)
x0{1} = (portf_v ./ value) .* w0{1};  % asset units (portfolio 1)
x0{2} = (portf_v ./ value) .* w0{2};  % asset units (portfolio 2)

% Quantile levels (99%, 99.9%)
alphas = [0.99 0.999];

% Compute VaR and CVaR (non-Normal and Normal) for 100000 scenarios
for(portN = 1:2)
    totloss = Losses_out * cell2mat(x0(portN));
    totloss = sort(totloss);
    meanloss = mu_l' * cell2mat(x0(portN));
    for(q=1:length(alphas))
        alf = alphas(q);
        % -------- Insert your code here -------- %
        VaRout(portN,q)  = totloss(ceil(Nout * alf));
        VaRinN(portN,q)  = meanloss + norminv(alf,0,1)*std(totloss);
        CVaRout(portN,q) = (1/(Nout*(1-alf))) * ( (ceil(Nout*alf)-Nout*alf) * VaRout(portN,q) + sum(totloss(ceil(Nout*alf)+1:Nout)) );
        CVaRinN(portN,q) = meanloss + (normpdf(norminv(alf,0,1))/(1-alf))*std(totloss);
        % -------- Insert your code here -------- %        
    end
end


% Perform 100 trials
N_trials = 100;
l1 = [];
portfMC1 = [];
portfMC2 = [];
for(tr=1:N_trials)
    
    % Monte Carlo approximation 1

    % -------- Insert your code here -------- %
    
    
    for s = 1:ceil(Nin/Ns) % systemic scenarios
        % -------- Insert your code here -------- %
        normcorr = sqrt_rho *randn(50,1);
        
        for si = 1:Ns % idiosyncratic scenarios for each systemic
            % -------- Insert your code here -------- %
            l1 = [];
            
            for i = 1:K               
                norm = normrnd(0,1);
                stddev = sqrt(1-(beta(i)^2));
                credit = (beta(i) * normcorr(driver(i))) + (stddev * norm);

                for c = 1:size(CS_Bdry,2)
                    credBound = CS_Bdry(i, :);
                    if(credit < credBound(c))
                        lossIndex = c;
                        break                
                    else
                        lossIndex = 7;
                    end            

                end

                loss = exposure(i,lossIndex);
                l1 = [l1; loss];           
            end
            Losses_inMC1((s - 1) * si + si, :) = l1;
        end
    end
    
    % Calculated losses for MC1 approximation (5000 x 100)
    
    
    % Monte Carlo approximation 2
    
    % -------- Insert your code here -------- %
  
    for s = 1:Nin % systemic scenarios (1 idiosyncratic scenario for each systemic)
        % -------- Insert your code here -------- %
       normcorr = sqrt_rho *randn(50,1);
        l2 = [];
        for i = 1:K               
            norm = normrnd(0,1);
            stddev = sqrt(1-(beta(i)^2));
            credit = (beta(i) * normcorr(driver(i))) + (stddev * norm);

            for c = 1:size(CS_Bdry,2)
                credBound = CS_Bdry(i, :);
                if(credit < credBound(c))
                    lossIndex = c;
                    break                
                else
                    lossIndex = 7;
                end            

            end

            loss = exposure(i,lossIndex);
            l2 = [l2; loss];           
        end
        
        Losses_inMC2(s, :) = l2;
    end
        
    % Calculated losses for MC2 approximation (5000 x 100)
   
    % Compute VaR and CVaR
    for(portN = 1:2)
        for(q=1:length(alphas))
            alf = alphas(q);
            % -------- Insert your code here -------- %            
            % Compute portfolio loss 
            portf_loss_inMC1 = sort(Losses_inMC1 * cell2mat(x0(portN)));
            portf_loss_inMC2 = sort(Losses_inMC2 * cell2mat(x0(portN)));
            mu_MCl = mean(Losses_inMC1)';
            var_MCl = cov(Losses_inMC1);
            mu_MC2 = mean(Losses_inMC2)';
            var_MC2 = cov(Losses_inMC2);
            % Compute portfolio mean loss mu_p_MC1 and portfolio standard deviation of losses sigma_p_MC1
            % Compute portfolio mean loss mu_p_MC2 and portfolio standard deviation of losses sigma_p_MC2
            % Compute VaR and CVaR for the current trial
            VaRinMC1{portN,q}(tr) = portf_loss_inMC1(ceil(Nin * alf));
            VaRinMC2{portN,q}(tr) = portf_loss_inMC2(ceil(Nin * alf));
            VaRinN1{portN,q}(tr) = mu_MCl' * cell2mat(x0(portN)) + norminv(alf,0,1)*std(portf_loss_inMC1);
            VaRinN2{portN,q}(tr) = mu_MC2' * cell2mat(x0(portN)) + norminv(alf,0,1)*std(portf_loss_inMC2);
            CVaRinMC1{portN,q}(tr) = (1/(Nin*(1-alf))) * ((ceil(Nin*alf)-Nin*alf) * VaRinMC1{portN,q}(tr) + sum(portf_loss_inMC1(ceil(Nin*alf)+1:Nin)));  
            CVaRinMC2{portN,q}(tr) = (1/(Nin*(1-alf))) * ((ceil(Nin*alf)-Nin*alf) * VaRinMC2{portN,q}(tr) + sum(portf_loss_inMC2(ceil(Nin*alf)+1:Nin)));  
            CVaRinN1{portN,q}(tr) = mu_MCl' * cell2mat(x0(portN)) + (normpdf(norminv(alf,0,1))/(1-alf))*std(portf_loss_inMC1);
            CVaRinN2{portN,q}(tr) = mu_MC2' * cell2mat(x0(portN)) + (normpdf(norminv(alf,0,1))/(1-alf))*std(portf_loss_inMC2);
            % -------- Insert your code here -------- %
            
        end
    end
  
end


% Display portfolio VaR and CVaR
for(portN = 1:2)
fprintf('\nPortfolio %d:\n\n', portN)    
 for(q=1:length(alphas))
    alf = alphas(q);
    fprintf('Out-of-sample: VaR %4.1f%% = $%6.2f, CVaR %4.1f%% = $%6.2f\n', 100*alf, VaRout(portN,q), 100*alf, CVaRout(portN,q))
    fprintf('In-sample MC1: VaR %4.1f%% = $%6.2f, CVaR %4.1f%% = $%6.2f\n', 100*alf, mean(VaRinMC1{portN,q}), 100*alf, mean(CVaRinMC1{portN,q}))
    fprintf('In-sample MC2: VaR %4.1f%% = $%6.2f, CVaR %4.1f%% = $%6.2f\n', 100*alf, mean(VaRinMC2{portN,q}), 100*alf, mean(CVaRinMC2{portN,q}))
    fprintf(' In-sample No: VaR %4.1f%% = $%6.2f, CVaR %4.1f%% = $%6.2f\n', 100*alf, VaRinN(portN,q), 100*alf, CVaRinN(portN,q))
    fprintf(' In-sample N1: VaR %4.1f%% = $%6.2f, CVaR %4.1f%% = $%6.2f\n', 100*alf, mean(VaRinN1{portN,q}), 100*alf, mean(CVaRinN1{portN,q}))
    fprintf(' In-sample N2: VaR %4.1f%% = $%6.2f, CVaR %4.1f%% = $%6.2f\n\n', 100*alf, mean(VaRinN2{portN,q}), 100*alf, mean(CVaRinN2{portN,q}))
 end
end

port1mc1loss = Losses_inMC1 * cell2mat(x0(1));
port1mc2loss = Losses_inMC1 * cell2mat(x0(2));
port2mc1loss = Losses_inMC2 * cell2mat(x0(1));
port2mc2loss = Losses_inMC2 * cell2mat(x0(2));
port1lossout = Losses_out * cell2mat(x0(1));
port2lossout = Losses_out * cell2mat(x0(2));
% Plot results
figure(1);
set(gcf, 'color', 'white');
[frequencyCounts, binLocations] = hist(sort(port1mc1loss), 100);
bar(binLocations, frequencyCounts);
hold on;
line([mean(VaRinMC1{1,1}) mean(VaRinMC1{1,1})], [0 max(frequencyCounts)/2], 'Color', 'r', 'LineWidth', 1, 'LineStyle', '--');
hold on;
normf = ( 1/(std(port1mc1loss)*sqrt(2*pi)) ) * exp( -0.5*((binLocations-mean(port1mc1loss))/std(port1mc1loss)).^2 );
normf = normf * sum(frequencyCounts)/sum(normf);
plot(binLocations, normf, 'r', 'LineWidth', 3);
hold on;
line([mean(VaRinMC1{1,2}) mean(VaRinMC1{1,2})], [0 max(frequencyCounts)/2], 'Color', 'r', 'LineWidth', 1, 'LineStyle', '-.');
line([mean(VaRinN1{1,1}) mean(VaRinN1{1,1})], [0 max(frequencyCounts)/2], 'Color', 'r', 'LineWidth', 1, 'LineStyle', '-.');
line([mean(VaRinN1{1,2}) mean(VaRinN1{1,2})], [0 max(frequencyCounts)/2], 'Color', 'r', 'LineWidth', 1, 'LineStyle', '-.');
hold off;
text(0.98*mean(VaRinMC1{1,1}), max(frequencyCounts)/1.9, 'VaR 99%')
text(0.98*mean(VaRinMC1{1,2}), max(frequencyCounts)/1.9, 'VaR 99.9%')
text(0.98*mean(VaRinN1{1,1}), max(frequencyCounts)/1.9, 'VaRn 99%')
text(0.98*mean(VaRinN1{1,2}), max(frequencyCounts)/1.9, 'VaRn 99.9%')
title('Portfolio 1 Loss Distribution, MC1')
xlabel('Portfolio 1 Loss')
ylabel('Frequency')
figure(2);
% -------- Insert your code here -------- %
set(gcf, 'color', 'white');
[frequencyCounts, binLocations] = hist(sort(port1mc2loss), 100);
bar(binLocations, frequencyCounts);
hold on;
line([mean(VaRinMC1{2,1}) mean(VaRinMC1{2,1})], [0 max(frequencyCounts)/2], 'Color', 'r', 'LineWidth', 1, 'LineStyle', '--');
hold on;
normf = ( 1/(std(port1mc2loss)*sqrt(2*pi)) ) * exp( -0.5*((binLocations-mean(port1mc2loss))/std(port1mc2loss)).^2 );
normf = normf * sum(frequencyCounts)/sum(normf);
plot(binLocations, normf, 'r', 'LineWidth', 3);
hold on;
line([mean(VaRinMC1{2,2}) mean(VaRinMC1{2,2})], [0 max(frequencyCounts)/2], 'Color', 'r', 'LineWidth', 1, 'LineStyle', '-.');
line([mean(VaRinN1{2,1}) mean(VaRinN1{2,1})], [0 max(frequencyCounts)/2], 'Color', 'r', 'LineWidth', 1, 'LineStyle', '-.');
line([mean(VaRinN1{2,2}) mean(VaRinN1{2,2})], [0 max(frequencyCounts)/2], 'Color', 'r', 'LineWidth', 1, 'LineStyle', '-.');
hold off;
text(0.98*mean(VaRinMC1{2,1}), max(frequencyCounts)/1.9, 'VaR 99%')
text(0.98*mean(VaRinMC1{2,2}), max(frequencyCounts)/1.9, 'VaR 99.9%')
text(0.98*mean(VaRinN1{2,1}), max(frequencyCounts)/1.9, 'VaRn 99%')
text(0.98*mean(VaRinN1{2,2}), max(frequencyCounts)/1.9, 'VaRn 99.9%')
title('Portfolio 2 Loss Distribution, MC1')
xlabel('Portfolio 2 Loss')
ylabel('Frequency')

figure(3);
set(gcf, 'color', 'white');
[frequencyCounts, binLocations] = hist(sort(port2mc1loss), 100);
bar(binLocations, frequencyCounts);
hold on;
line([mean(VaRinMC2{1,1}) mean(VaRinMC2{1,1})], [0 max(frequencyCounts)/2], 'Color', 'r', 'LineWidth', 1, 'LineStyle', '--');
hold on;
normf = ( 1/(std(port2mc1loss)*sqrt(2*pi)) ) * exp( -0.5*((binLocations-mean(port2mc1loss))/std(port2mc1loss)).^2 );
normf = normf * sum(frequencyCounts)/sum(normf);
plot(binLocations, normf, 'r', 'LineWidth', 3);
hold on;
line([mean(VaRinMC2{1,2}) mean(VaRinMC2{1,2})], [0 max(frequencyCounts)/2], 'Color', 'r', 'LineWidth', 1, 'LineStyle', '-.');
line([mean(VaRinN2{1,1}) mean(VaRinN2{1,1})], [0 max(frequencyCounts)/2], 'Color', 'r', 'LineWidth', 1, 'LineStyle', '-.');
line([mean(VaRinN2{1,2}) mean(VaRinN2{1,2})], [0 max(frequencyCounts)/2], 'Color', 'r', 'LineWidth', 1, 'LineStyle', '-.');
hold off;
text(0.98*mean(VaRinMC2{1,1}), max(frequencyCounts)/1.9, 'VaR 99%')
text(0.98*mean(VaRinMC2{1,2}), max(frequencyCounts)/1.9, 'VaR 99.9%')
text(0.98*mean(VaRinN2{1,1}), max(frequencyCounts)/1.9, 'VaRn 99%')
text(0.98*mean(VaRinN2{1,2}), max(frequencyCounts)/1.9, 'VaRn 99.9%')
title('Portfolio 1 Loss Distribution, MC2')
xlabel('Portfolio 1 Loss')
ylabel('Frequency')

figure(4);
set(gcf, 'color', 'white');
[frequencyCounts, binLocations] = hist(sort(port2mc2loss), 100);
bar(binLocations, frequencyCounts);
hold on;
line([mean(VaRinMC2{2,1}) mean(VaRinMC2{2,1})], [0 max(frequencyCounts)/2], 'Color', 'r', 'LineWidth', 1, 'LineStyle', '--');
hold on;
normf = ( 1/(std(port2mc2loss)*sqrt(2*pi)) ) * exp( -0.5*((binLocations-mean(port2mc2loss))/std(port2mc2loss)).^2 );
normf = normf * sum(frequencyCounts)/sum(normf);
plot(binLocations, normf, 'r', 'LineWidth', 3);
hold on;
line([mean(VaRinMC2{2,2}) mean(VaRinMC2{2,2})], [0 max(frequencyCounts)/2], 'Color', 'r', 'LineWidth', 1, 'LineStyle', '-.');
line([mean(VaRinN2{2,1}) mean(VaRinN2{2,1})], [0 max(frequencyCounts)/2], 'Color', 'r', 'LineWidth', 1, 'LineStyle', '-.');
line([mean(VaRinN2{2,2}) mean(VaRinN2{2,2})], [0 max(frequencyCounts)/2], 'Color', 'r', 'LineWidth', 1, 'LineStyle', '-.');
hold off;
text(0.98*mean(VaRinMC2{2,1}), max(frequencyCounts)/2.5, 'VaR 99%')
text(0.98*mean(VaRinMC2{2,2}), max(frequencyCounts)/1.9, 'VaR 99.9%')
text(0.98*mean(VaRinN2{2,1}), max(frequencyCounts)/1.9, 'VaRn 99%')
text(0.98*mean(VaRinN2{2,2}), max(frequencyCounts)/1.9, 'VaRn 99.9%')
title('Portfolio 2 Loss Distribution, MC2')
xlabel('Portfolio 2 Loss')
ylabel('Frequency')

figure(5);
set(gcf, 'color', 'white');
[frequencyCounts, binLocations] = hist(sort(port1lossout), 100);
bar(binLocations, frequencyCounts);
hold on;
line([VaRout(1,1) VaRout(1,1)], [0 max(frequencyCounts)/2], 'Color', 'r', 'LineWidth', 1, 'LineStyle', '--');
hold on;
normf = ( 1/(std(port1lossout)*sqrt(2*pi)) ) * exp( -0.5*((binLocations-mean(port1lossout))/std(port1lossout)).^2 );
normf = normf * sum(frequencyCounts)/sum(normf);
plot(binLocations, normf, 'r', 'LineWidth', 3);
hold on;
line([VaRout(1,2) VaRout(1,2)], [0 max(frequencyCounts)/2], 'Color', 'r', 'LineWidth', 1, 'LineStyle', '-.');
line([VaRinN(1,1) VaRinN(1,1)], [0 max(frequencyCounts)/2], 'Color', 'r', 'LineWidth', 1, 'LineStyle', '-.');
line([VaRinN(1,2) VaRinN(1,2)], [0 max(frequencyCounts)/2], 'Color', 'r', 'LineWidth', 1, 'LineStyle', '-.');
hold off;
text(0.98*VaRout(1,1), max(frequencyCounts)/1.9, 'VaR 99%')
text(0.98*VaRout(1,2), max(frequencyCounts)/1.9, 'VaR 99.9%')
text(0.98*VaRinN(1,1), max(frequencyCounts)/1.9, 'VaRn 99%')
text(0.98*VaRinN(1,2), max(frequencyCounts)/1.9, 'VaRn 99.9%')
title('Portfolio Losses Out of Sample, Portfolio 1')
xlabel('Portfolio Loss')
ylabel('Frequency')

figure(6);
set(gcf, 'color', 'white');
[frequencyCounts, binLocations] = hist(sort(port2lossout), 100);
bar(binLocations, frequencyCounts);
hold on;
line([VaRout(2,1) VaRout(2,1)], [0 max(frequencyCounts)/2], 'Color', 'r', 'LineWidth', 1, 'LineStyle', '--');
hold on;
normf = ( 1/(std(port2lossout)*sqrt(2*pi)) ) * exp( -0.5*((binLocations-mean(port2lossout))/std(port2lossout)).^2 );
normf = normf * sum(frequencyCounts)/sum(normf);
plot(binLocations, normf, 'r', 'LineWidth', 3);
hold on;
line([VaRout(2,2) VaRout(2,2)], [0 max(frequencyCounts)/2], 'Color', 'r', 'LineWidth', 1, 'LineStyle', '-.');
line([VaRinN(2,1) VaRinN(2,1)], [0 max(frequencyCounts)/2], 'Color', 'r', 'LineWidth', 1, 'LineStyle', '-.');
line([VaRinN(2,2) VaRinN(2,2)], [0 max(frequencyCounts)/2], 'Color', 'r', 'LineWidth', 1, 'LineStyle', '-.');
hold off;
text(0.98*VaRout(2,1), max(frequencyCounts)/2.5, 'VaR 99%')
text(0.98*VaRout(2,2), max(frequencyCounts)/1.9, 'VaR 99.9%')
text(0.98*VaRinN(2,1), max(frequencyCounts)/1.9, 'VaRn 99%')
text(0.98*VaRinN(2,2), max(frequencyCounts)/1.9, 'VaRn 99.9%')
title('Portfolio Losses Out of Sample, Portfolio 2')
xlabel('Portfolio Loss')
ylabel('Frequency')

figure(7)
qqplot(sort(port1lossout));
title('QQ Plot, Out of Sample Portfolio 1')
xlabel('Quantile Normal')
ylabel('Quantile Sample')

figure(8)
qqplot(sort(port2lossout));
title('QQ Plot, Out of Sample Portfolio 2')
xlabel('Quantile Normal')
ylabel('Quantile Sample')