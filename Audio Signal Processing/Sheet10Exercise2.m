%% Exercise 10, 2.a

% build a filter H with an order 20
bpFilt_H20 = designfilt('bandpassfir','FilterOrder',20, ...
         'CutoffFrequency1',200,'CutoffFrequency2',400, ...
         'SampleRate',1500);

% plot the constructed filter
% fvtool(bpFilt)

% get the frequency response
[frequency_response, frequency_range]= freqz(bpFilt_H20);



% plot the frequency response of bpFilt_H20(Filter h with a degree of 20)
subplot (5,2,1);
plot(frequency_range/pi,abs(frequency_response)); title('Amplitude response of the filter h'); grid on; xlabel('Frequency'); ylabel('Magnitude');

subplot(5,2,2);
plot(frequency_range/pi,angle(frequency_response)/pi); title('Phase response of the filter h'); grid on; xlabel('Frequency'); ylabel('Phase/pi');

%% Exercise 10, 2.b

% build a filter H with an order 10
bpFilt_H10 = designfilt('bandpassfir','FilterOrder',10, ...
         'CutoffFrequency1',200,'CutoffFrequency2',400, ...
         'SampleRate',1500);
[frequency_response, frequency_range]= freqz(bpFilt_H10);

%fvtool(bpFilt_H100)

% plot the frequency response of bpFilt_H10(Filter h with a degree of 20)
subplot (5,2,3);
plot(frequency_range/pi,abs(frequency_response)); title('Amplitude response of the filter h10'); grid on; xlabel('Frequency'); ylabel('Magnitude');

subplot(5,2,4);
plot(frequency_range/pi,angle(frequency_response)/pi); title('Phase response of the filter h10'); grid on; xlabel('Frequency'); ylabel('Phase/pi');

     
 % build a filter H with an order 50
bpFilt_H50 = designfilt('bandpassfir','FilterOrder',50, ...
         'CutoffFrequency1',200,'CutoffFrequency2',400, ...
         'SampleRate',1500);
[frequency_response, frequency_range]= freqz(bpFilt_H50);
     
% plot the frequency response of bpFilt_H20(Filter h with a degree of 20)
subplot (5,2,5);
plot(frequency_range/pi,abs(frequency_response)); title('Amplitude response of the filter h50'); grid on; xlabel('Frequency'); ylabel('Magnitude');

subplot(5,2,6);
plot(frequency_range/pi,angle(frequency_response)/pi); title('Phase response of the filter h50'); grid on; xlabel('Frequency'); ylabel('Phase/pi');

     
% build a filter H with an order 100
bpFilt_H100 = designfilt('bandpassfir','FilterOrder',100, ...
         'CutoffFrequency1',200,'CutoffFrequency2',400, ...
         'SampleRate',1500);
[frequency_response, frequency_range]= freqz(bpFilt_H100);

% plot the frequency response of bpFilt_H20(Filter h with a degree of 20)
subplot (5,2,7);
plot(frequency_range/pi,abs(frequency_response)); title('Amplitude response of the filter h100'); grid on; xlabel('Frequency'); ylabel('Magnitude');

subplot(5,2,8);
plot(frequency_range/pi,angle(frequency_response)/pi); title('Phase response of the filter h100'); grid on; xlabel('Frequency'); ylabel('Phase/pi');

%% Exercise 10, 2.c 
 
sample_rate= 8000; % sampling rate=8khz 
time = (0:1/sample_rate:1);
frequency1 = 300; % frequency = 300hz
frequency2 = 1000; % frequency = 1000hz

% create signal x
x = sin(2*pi*time*frequency1) + sin(2*pi*time*frequency2);
fft_x = fft(x) % calculate fourier transform of x

%plot the fourier transform of the signal x
subplot(5,2,9);
plot(abs(fft_x)); title('Fourier transform of the input signal x'); grid on; xlabel('frequency'); ylabel('maginute');


y = filter(bpFilt_20, x); % apply filter and get output y
fft_y = fft(y) % calculate fourier transform of y

%plot the fourier transform of the filtered signal y
subplot(5,2,10);
plot(abs(fft_y)); title('Fourier transform of the filtered signal y'); grid on; xlabel('frequency'); ylabel('maginute');

%plays the signal x first, then after 3 seconds plays y 
soundsc(x);
pause(3);
soundsc(y);

   
     
