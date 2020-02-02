%% Kevin Mahabeer | ECET 480 | Homework 1
clear all; close all; clc;
%% 1.
%{
The minimum carrier frequency is the bandwidth of the message signal.
%}

%% 2.
%{
Generate AM and FM signals (AM index = 0.35 and FM index of 0.4, 1, 2). 
Compare the spectra and obtain the bandwidths in each case. Use a single 
tone signal of 2 KHz. Choose an appropriate carrier frequency. 
[Examples available in notes].
%}

%% 2. Generate Signals
clear; clc; 

fm = 2e3; % message frequency
fc = 2*fm; % carrier frequency
fs = fc+20*fm; % sampling frequency

Ts = 1/fs; % period
t = 0:Ts:0.005; % time vector

% Single tone message signal (fm = 2kHz)
m = sin(2*pi*fm*t);

% Carrier Signal
A0 = 1; % carrier amplitude
c = A0*cos(2*pi*fc*t); 

% Amplitude Modulated Signal
ka = 0.35;
sAM = A0*(1+ka.*m).*cos(2*pi*fc*t);

% Frequency Modulated Signal
% FM index = 0.4
kf1 = 0.4;
sFM1 = A0*cos(2*pi*(fc.*t)+2*pi*kf1*cumsum(m));

% FM index = 1
kf2 = 1;
sFM2 = A0*cos(2*pi*(fc.*t)+2*pi*kf2*cumsum(m));

% FM index = 2
kf3 = 2;
sFM3 = A0*cos(2*pi*(fc.*t)+2*pi*kf3*cumsum(m));

%% 2. Plot signals

% Plot message and carrier signals
figure;
subplot(2,3,1)
plot(t,m,'Linewidth',1)
grid on; xlim([0 t(end)]); ylim([-1.5 1.5]);
xlabel('Time'); %ylabel('Message Signal')
title(['Single Tone Message Signal, f_m = ',num2str(fm)])

subplot(2,3,2)
plot(t,c,'Linewidth',1)
grid on; xlim([0 t(end)]); ylim([-1.5 1.5]);
xlabel('Time'); %ylabel('Carrier Signal')
title('Carrier Signal')

% Plot Amplitude Modulation
subplot(2,3,3)
plot(t,sAM,'Linewidth',1)
grid on; xlim([0 t(end)]); ylim([-1.5 1.5]);
xlabel('Time'); %ylabel('AM Signal')
title(['AM Signal, k_a = ',num2str(ka)])

% 2. Plot Frequency Modulation
subplot(2,3,4)
plot(t,sFM1,'Linewidth',1)
grid on; xlim([0 t(end)]); ylim([-1.5 1.5]);
xlabel('Time'); %ylabel('FM Signal')
title(['FM Signal, k_f = ',num2str(kf1)])

subplot(2,3,5)
plot(t,sFM2,'Linewidth',1)
grid on;  xlim([0 t(end)]); ylim([-1.5 1.5]);
xlabel('Time'); %ylabel('FM Signal')
title(['FM Signal, k_f = ',num2str(kf2)])

subplot(2,3,6)
plot(t,sFM3,'Linewidth',1)
grid on; xlim([0 t(end)]); ylim([-1.5 1.5]);
xlabel('Time'); %ylabel('FM Signal')
title(['FM Signal, k_f = ',num2str(kf3)])

%% 2. Spectrum of message signal

N = length(m); % length of message signal (Same for carrier and AM, FM)
n = 6;
W=[0.05,0.1,0.15,0.2];
Ws = 2.*pi/Ts; % Angular frequency resolution (rad/s)
fMax = 0.5*Ws/(2*pi);
Ww = Ws*(0:N/2.)/N; % Angular frequency (rad/s)
Wf = (1/(2*pi))*Ww;

% Message Spectra
FBm = fft(m); % FFT
FBPm = FBm(1:N/2+1)*Ts;
FBm = FBPm/max(abs(FBPm));

% AM Spectra
FBa = fft(sAM); % FFT
FBPa = FBa(1:N/2+1)*Ts;
FBa = FBPa/max(abs(FBPa));

% FM1 Spectra
FBf1 = fft(sFM1); % FFT
FBPf1 = FBf1(1:N/2+1)*Ts;
FBf1 = FBPf1/max(abs(FBPf1));

% FM2 Spectra
FBf2 = fft(sFM2); % FFT
FBPf2 = FBf2(1:N/2+1)*Ts;
FBf2 = FBPf2/max(abs(FBPf2));

%% Plot signal v. spectrum of message and AM
% Message v. Message Spectra
figure;
subplot(2,2,1)
plot(t,m,'Linewidth',1)
grid on; xlim([0 t(end)]); ylim([-1.5 1.5]);
xlabel('Time'); %ylabel('Message Signal')
title('Single Tone Message Signal')

subplot(2,2,2)
plot(Wf,20*log10(abs(FBm)))
grid on; axis tight; %xlim([0 t(end)]); ylim([-1.5 1.5]);
xlabel('Time'); %ylabel('Message Signal')
title('Single Tone Message Signal Spectra')

% AM v. AM Spectra
subplot(2,2,3)
plot(t,sAM,'Linewidth',1)
grid on; xlim([0 t(end)]); ylim([-1.5 1.5]);
xlabel('Time'); %ylabel('Message Signal')
title(['AM Signal, f_m = ',num2str(fm),'Hz'])

subplot(2,2,2)
plot(Wf,20*log10(abs(FBm)))
grid on; axis tight; %xlim([0 t(end)]); ylim([-1.5 1.5]);
xlabel('Time'); %ylabel('Message Signal')
title('Message Signal Spectra')

% AM v. AM Spectra
subplot(2,2,4)
plot(Wf,20*log10(abs(FBa)))
grid on; axis tight; %xlim([0 t(end)]); ylim([-1.5 1.5]);
xlabel('Time'); %ylabel('Message Signal')
title(['AM Signal Spectra, k_a = ',num2str(ka)])

%% Plot signal v. spectrum of message and FM
% Message v. Message Spectra
figure;
subplot(3,2,1)
plot(t,m,'Linewidth',1)
grid on; xlim([0 t(end)]); ylim([-1.5 1.5]);
xlabel('Time'); %ylabel('Message Signal')
title('Single Tone Message Signal')

subplot(3,2,2)
plot(Wf,20*log10(abs(FBm)))
grid on; axis tight; %xlim([0 t(end)]); ylim([-1.5 1.5]);
xlabel('Time'); %ylabel('Message Signal')
title(['Single Tone Message Signal Spectra'])

% FM v FM Spectra f_k1
subplot(3,2,3)
plot(t,sFM1,'Linewidth',1)
grid on; xlim([0 t(end)]); ylim([-1.5 1.5]);
xlabel('Time'); %ylabel('FM Signal')
title(['FM Signal, k_f = ',num2str(kf1)])

subplot(3,2,4)
plot(Wf,20*log10(abs(FBf1)))
grid on; axis tight; %xlim([0 t(end)]); ylim([-1.5 1.5]);
xlabel('Time'); %ylabel('Message Signal')
title(['FM Signal Spectra, k_f = ',num2str(kf1)])

% FM v FM Spectra f_k2
subplot(3,2,5)
plot(t,sFM2,'Linewidth',1)
grid on; xlim([0 t(end)]); ylim([-1.5 1.5]);
xlabel('Time'); %ylabel('FM Signal')
title(['FM Signal, k_f = ',num2str(kf2)])

subplot(3,2,6)
plot(Wf,20*log10(abs(FBf2)))
grid on; axis tight; %xlim([0 t(end)]); ylim([-1.5 1.5]);
xlabel('Time'); %ylabel('Message Signal')
title(['FM Signal Spectra, k_f = ',num2str(kf2)])


%% 3.a) Low Pass Filter
clear;

n = 6;
W = [0.05, 0.1, 0.15, 0.2];

T = 1; % Pulse duration
fs = 32/T; % sampling frequency
Ts = 1/fs; % sampling period
N = 4096; % Number of points used for N-pt DFT
t = [0:N-1]*Ts; % N time samples

% Message Signal
m = abs(sinc(4.*(t-8*T))).*rectpuls(t-8*T,T);

% Plot original signal 
figure;
plot(t/T,m,'Linewidth',1.5)
grid on; hold on;
xlim([6 10]); ylim([-1.5 1.5]);
title('Single Tone Message signal with low pass filters')
xlabel('Time (ms)'); ylabel('Message Signal');

% Plot filtered signals
for k = 1:4
    [B,A] = butter(n,W(k));
    mOut = filtfilt(B,A,m);
    plot(t/T,mOut,'--')
    grid on; hold on;
    xlim([6 10]);
    ylim([-1.5 1.5]);
end

legend('input',...
    ['W_n=',num2str(W(1)),', f_c=',num2str(W(1)*fs*1e-6/2),'MHz'],... 
    ['W_n=',num2str(W(2)),', f_c=',num2str(W(2)*fs*1e-6/2),'MHz'],... 
    ['W_n=',num2str(W(3)),', f_c=',num2str(W(3)*fs*1e-6/2),'MHz'],... 
    ['W_n=',num2str(W(4)),', f_c=',num2str(W(4)*fs*1e-6/2),'MHz'],... 
    'location','best');

hold off;

%% 3.b) Modulation

T = 1e-3; % pulse duration
fm = 1/T;
f0 = 10/T; 
fs = 32/T; %sampling rate
Ts = 1/fs; % sampling interval

% Carrier Signal
A0 = 1;
c = A0*cos(2*pi*f0*t);

% Amplitude Modulated Signal
ka = 0.35;
sAM = A0*(1+ka.*m).*cos(2*pi*f0*t);

% Frequency Modulated Signal
kf = 0.7;
sFM4 = A0*cos(2*pi*(f0.*t)+2*pi*kf*cumsum(m));

% Plot signals
figure;
subplot(2,2,1)
plot(t,m,'Linewidth',2)
grid on; xlim([7 9]); ylim([-1.5 1.5]);
xlabel('Time'); ylabel('Message Signal')
title('Message Signal')

subplot(2,2,2)
plot(t,c,'Linewidth',1)
grid on; xlim([7 9]); ylim([-1.5 1.5]);
xlabel('Time'); ylabel('Carrier Signal')
title('Carrier Signal')

subplot(2,2,3)
plot(t,sAM,'Linewidth',1)
grid on; xlim([7 9]); ylim([-1.5 1.5]);
xlabel('Time'); ylabel('AM Signal')
title('AM Signal')

subplot(2,2,4)
plot(t,sFM4,'Linewidth',1)
grid on; xlim([7 9]); ylim([-1.5 1.5]);
xlabel('Time'); ylabel('FM Signal')
title('FM Signal')

%% 3.b) Spectrum (no noise)
N = length(m); % length of message signal (Same for carrier and AM, FM)

n = 6;
W = [0.05,0.1,0.15,0.2];
Ws = 2.*pi/Ts; % Angular frequency resolution (rad/s)
fMax = 0.5*Ws/(2*pi);
Ww = Ws*(0:N/2.)/N; % Angular frequency (rad/s)
Wf = (1/(2*pi))*Ww;

% Message Spectra
FBm = fft(m); % FFT
FBPm = FBm(1:N/2+1)*Ts;
FBm = FBPm/max(abs(FBPm));

% Carrier Spectra
FBc = fft(c); % FFT
FBPc = FBc(1:N/2+1)*Ts;
FBc = FBPc/max(abs(FBPc));

% AM Spectra
FBa = fft(sAM); % FFT
FBPa = FBa(1:N/2+1)*Ts;
FBa = FBPa/max(abs(FBPa));

% FM1 Spectra
FBf1 = fft(sFM4); % FFT
FBPf1 = FBf1(1:N/2+1)*Ts;
FBf1 = FBPf1/max(abs(FBPf1));

%% 3.b) Plot (no noise)

figure;
subplot(2,2,1)
plot(Wf,20*log10(abs(FBm)))
grid on; axis tight; %xlim([0 t(end)]); ylim([-1.5 1.5]);
xlabel('Time'); %ylabel('Message Signal')
title('Message Signal Spectra')

subplot(2,2,2)
plot(Wf,20*log10(abs(FBc)))
grid on; axis tight; %xlim([0 t(end)]); ylim([-1.5 1.5]);
xlabel('Time'); %ylabel('Message Signal')
title('Carrier Signal Spectra')

subplot(2,2,3)
plot(Wf,20*log10(abs(FBa)))
grid on; axis tight; %xlim([0 t(end)]); ylim([-1.5 1.5]);
xlabel('Time'); %ylabel('Message Signal')
title(['AM Signal Spectra, k_a = ',num2str(ka)])

subplot(2,2,4)
plot(Wf,20*log10(abs(FBf1)))
grid on; axis tight; %xlim([0 t(end)]); ylim([-1.5 1.5]);
xlabel('Time'); %ylabel('Message Signal')
title(['FM Signal Spectra, k_f = ',num2str(kf)])

%% 3.b) Demodulation (no noise)

figure;
subplot(221)
plot(t,m,'Linewidth',2)
grid on; xlim([7 9]); ylim([-1.5 1.5]);
xlabel('Time'); ylabel('Message Signal')
title('Original Signal')

subplot(222)
demodAM = 6*(sAM.*cos(t*f0));
[B,A] = butter(20, .422, 'low');
filteredAM = filtfilt(B,A,demodAM);
plot(t,filteredAM,'Linewidth',2)
grid on; xlim([7 9]); ylim([-1.5 1.5]);
xlabel('Time'); ylabel('Message Signal')
title('Detected AM signal')

subplot(223)
der = [0, diff(sFM4)];
envelopFM = 2*abs(der);
[B,A] = butter(25, .7, 'low');
filteredFM = filtfilt(B,A,envelopFM);
plot(t,filteredFM,'Linewidth',2)
grid on; xlim([7 9]); ylim([-1.5 1.5]);
xlabel('Time'); ylabel('Message Signal')
title({'Detected FM signal','(Differentiator)'})

subplot(224)
hill = hilbert(sFM4);
rHill = unwrap(angle(hill.*exp(1i*2*pi*f0*t)));
rHill = [0, diff(rHill)];
[B,A] = butter(25, .7, 'low');
filteredHillFM = filtfilt(B,A,rHill);
plot(t,filteredHillFM,'Linewidth',2)
grid on; xlim([7 9]); ylim([-1.5 1.5]);
xlabel('Time'); ylabel('Message Signal')
title({'Detected FM signal','(using pre-envelope)'})

%% 3.b) Adding noise
clear;

n = 6;
W = [0.05, 0.1, 0.15, 0.2];

T = 1; % Pulse duration
fs = 32/T; % sampling frequency
Ts = 1/fs; % sampling period
N = 4096; % Number of points used for N-pt DFT

t = [0:N-1]*Ts; % N time samples

m = abs(sinc(4.*(t-8*T))).*(rectpuls(t-8*T,T)+0.2*randn(1,N));

T = 1e-3; % pulse duration
% fm = 1/T;
f0 = 10/T;
fs = 32/T; %sampling rate
Ts = 1/fs; % sampling interval

% Carrier Signal
A0 = 1;
c = A0*cos(2*pi*f0*t);

% Amplitude Modulated Signal
ka = 0.35;
sAM = A0*(1+ka.*m).*cos(2*pi*f0*t);

% Frequency Modulated Signal
kf = 0.7;
sFM4 = A0*cos(2*pi*(f0.*t)+2*pi*kf*cumsum(m));

figure;
subplot(2,2,1)
plot(t,m,'Linewidth',2)
grid on; xlim([7 9]); ylim([-1.5 1.5]);
xlabel('Time'); ylabel('Message Signal')
title('Message Signal')

subplot(2,2,2)
plot(t,c,'Linewidth',1.5)
grid on; xlim([7 9]); ylim([-1.5 1.5]);
xlabel('Time'); ylabel('Carrier Signal')
title('Carrier Signal')

subplot(2,2,3)
plot(t,sAM,'Linewidth',1.5)
grid on; xlim([7 9]); ylim([-1.5 1.5]);
xlabel('Time'); ylabel('AM Signal')
title('AM Signal')

subplot(2,2,4)
plot(t,sFM4,'Linewidth',0.25)
grid on; xlim([7 9]); ylim([-1.5 1.5]);
xlabel('Time'); ylabel('FM Signal')
title('FM Signal')

%% Demodulate Signal With Noise

figure;
subplot(221)
plot(t,m,'Linewidth',2)
grid on; xlim([7 9]); ylim([-1.5 1.5]);
xlabel('Time'); ylabel('Message Signal')
title('Input Signal')

subplot(222)
demodAM = 6*(sAM.*cos(t*f0));
[B,A] = butter(20, .422, 'low');
filteredAM = filtfilt(B,A,demodAM);
plot(t,filteredAM,'Linewidth',2)
grid on; xlim([7 9]); ylim([-1.5 1.5]);
xlabel('Time'); ylabel('Message Signal')
title('Detected AM Signal')

subplot(223)
der = [0, diff(sFM4)];
envelopFM = 2*abs(der);
[B,A] = butter(25, .7, 'low');
filteredFM = filtfilt(B,A,envelopFM);
plot(t,filteredFM,'Linewidth',2)
grid on; xlim([7 9]); ylim([-1.5 1.5]);
xlabel('Time'); ylabel('Message Signal')
title({'Detected FM signal','(Differentiator)'})

subplot(224)
hill = hilbert(sFM4);
rHill = unwrap(angle(hill.*exp(1i*2*pi*f0*t)));
rHill = [0, diff(rHill)];
[B,A] = butter(25, .7, 'low');
filteredHillFM = filtfilt(B,A,rHill);
plot(t,filteredHillFM,'Linewidth',2)
grid on; xlim([7 9]); ylim([-1.5 1.5]);
xlabel('Time'); ylabel('Message Signal')
title({'Detected FM signal','(using pre-envelope)'})


