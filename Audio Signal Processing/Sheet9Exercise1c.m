% time domain range [-300, 300]
time = -15:15;
X = zeros(1, length(time)); X_min = 1; X_max =4;
Y = zeros(1, length(time)); Y_min=1; Y_max=4;

%x = (x(0); x(1); x(2); x(3)) := (1; 1; 1; 1) and y = (y(0); y(1); y(2); y(3)) := (1; 1; 1; 1).
x = [1,1,1,1]; y = [1,1,1,1];
X(time>=X_min&time<=X_max) = x;
Y(time>=Y_min&time<=Y_max) = y;

conv_min=X_min+Y_min; conv_max=X_max+Y_max;


plot_convolution(time,X,Y,conv_min,conv_max);


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
    
    
    pause(0.1);
    
end

end