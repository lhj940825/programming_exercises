clc; clear all; close all;


t = -500:500; % Time range
X = zeros(1,length(t)); x1 = -150; x2 = 150; % Input signal and its range
H = zeros(1,length(t)); h1 = -50; h2 = 50; % Impulse responce and its range
Y = zeros(1,length(t)); xh1 = x1+h1; xh2 = x2+h2; % Convolution output
x = sin(pi*t/10)./(pi*t/10); x(t==0)=1; % Generate input signal
h = t./t; h(t==0)=1; % Generate impulse respose
% h = exp(-0. s002*t.^2); % Generate impulse respose
H(t>=h1&t<=h2) = h(t>=h1&t<=h2); % Fit the input signal within range
X(t>=x1&t<=x2) = x(t>=x1&t<=x2); % Fit the impulse response within range
% Plot original signal and impulse response
subplot(3,2,1); 
plot(t,X,'LineWidth',3); title('Input signal'); grid on;
subplot(3,2,2); 
plot(t,H,'r','LineWidth',3); title('Impulse responce'); grid on;
for n = xh1:xh2 % Convolution limits
    % Convolution steps
    f = fliplr(X);           % Step 1: Flip 
    Xm = circshift(f,[0,n]); % Step 2: Shift
    m = Xm.*H;               % Step 3: Multiply 
    Y(t==n) = sum(m);        % Step 4: add/integrate/sum
    
    % Convolution operation live
    subplot(3,2,[3 4]); 
    plot(t,H,'r',t,circshift(fliplr(X),[0,n]),'LineWidth',3); grid on;
    title('Convolution operation: Flip, Shift, Multiply, and add')
    
    % Result of convolution live
    subplot(3,2,[5 6]); 
    plot(t,Y,'m','LineWidth',3); axis tight; grid on;
    title('Convolution output')
    
    pause(0.01); % Pause for a while
end