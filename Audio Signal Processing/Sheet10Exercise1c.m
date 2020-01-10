%% Exercise 10, 1.c
clc; clear all; close all;

% set time domain range
time = -10:10;
H= zeros(1, length(time)); H_min=0; H_max=1;
G= zeros(1, length(time)); G_min=0; G_max=1;

%h = (h(0), h(1)) := (0.5, 0.5) and g = (g(0); g(1)) := (0.5; -0.5).
h = [0.5,0.5]; g = [0.5, -0.5];
H(time>=H_min&time<=H_max) = h;
G(time>=G_min&time<=G_max) = g;


% plot coefficients of the two filters h and g
subplot(4,2,1);
stem(time,H); title('coefficient of the filter h'); grid on;

subplot(4,2,2);
stem(time,G); title('coefficient of the filter g'); grid on;


% calculate the frequency, amplitude, and phase response of the filter h
[frequency_response, frequency_range]= freqz(H);
subplot(4,2,3);
plot(frequency_range/pi,abs(20*log10(abs(frequency_response)))); title('Amplitude response of the filter h'); grid on; xlabel('Frequency'); ylabel('Magnitude(DB)');

subplot(4,2,5);
plot(frequency_range/pi,angle(frequency_response)/pi); title('Phase response of the filter h'); grid on; xlabel('Frequency'); ylabel('Phase/pi');

% calculate the frequency, amplitude, and phase response of the filter g
[frequency_response, frequency_range]= freqz(G);
subplot(4,2,4);
plot(frequency_range/pi,abs(20*log10(abs(frequency_response)))); title('Amplitude response of the filter g'); grid on;xlabel('Frequency'); ylabel('Magnitude(DB)');

subplot(4,2,6);
plot(frequency_range/pi,angle(frequency_response)/pi); title('Phase response of the filter g'); grid on; xlabel('Frequency'); ylabel('Phase/pi');






