clc; clear all; close all;

% time domain range [-300, 300]
time = -300:300;
X = zeros(1, length(time)); X_min = -50; X_max =50;
Y = zeros(1, length(time)); Y_min=-25; Y_max=25;


% test input signal
x = cos(pi*time/20);

% build shifted delta function
y = dirac(time);
idx = y == Inf; % find the index of Infinite value and switch it to value 1
y(idx) = 1;
y = circshift(y,20);

% crop the signals and save into X(input) and Y(filter)
Y(time>=Y_min & time<=Y_max) = y(time>=Y_min & time<=Y_max);
X(time>=X_min & time<=X_max) = x(time>=X_min & time<=X_max)

conv_min = X_min+Y_min; conv_max = X_max+Y_max;

plot_convolution(time,X,Y,conv_min, conv_max);

% exercise 1.b
function []=plot_convolution(time, X, Y, conv_min, conv_max)
conv_output = zeros(1, length(time));

subplot(5,1,1);
plot(time,X,'LineWidth',2); title('Input Signal X'); grid on;

subplot(5,1,2);
plot(time,Y,'g','LineWidth',2); title('Filter Y'); grid on;

for n = conv_min:conv_max % convolution calculation range(index): [conv_min, conv_max]
    Y_flip = fliplr(Y); % flip function left to right: Y(k)-> Y(-k)
    Y_flip_shifted = circshift(Y_flip,n); % shift Y(-k) by n -> Y(n-k)
    mul_X_and_Y_flip_shifted = X.*Y_flip_shifted;
    sum_mul_X_and_Y_flip_shifted = sum(mul_X_and_Y_flip_shifted);
    conv_output(time==n) = sum_mul_X_and_Y_flip_shifted;
    
    
    subplot(5,1,3);
    plot(time, Y_flip_shifted,'g', 'LineWidth',2); title('Y(n-k)'); grid on;
    
    %plot the overlapped area by X(k) and Y(n-k)
    subplot(5,1,4);
    plot(time, Y_flip_shifted,'g', time, X,'LineWidth',3); title('X(k)*Y(n-k)'); grid on;
    
    %plot the convolution output figure
    subplot(5,1,5);
    plot(time, conv_output, 'r', 'LineWidth',3); title('convolution(X,Y)'); grid on;
    
    
    pause(0.01);
    
end

end