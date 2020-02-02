%% Kevin Mahabeer | ECET 480 | Homework

%% 1.
clear; close all; clc;

%{
Simulate the error rates in a coherent BPSK modem and compare your results
to the theoretical ones. For the noise below, obtain the error rate. Assume
that it is white in the bandwidth of interest.

The noise has a double sided inverse Gaussian with mu = 1.5 and lambda = 1.
If the variance is not unity, make the variance of the set unity.
%}

%% 1. Generate binary data, BPSK modulated signal, and BPSK + Noise
%{
% Manual way of creating noise
Double sided inverse Gaussian noise
pf1 = makedist('InverseGaussian','mu',1.5,'lambda',1);

x1 = linspace(0,8);
x2 = linspace(-8,0);

y1 = pdf(pf1,x1);
y2 = flip(y1);

n = [y2, y1];

figure('Name','Noise');
plot([x2,x1], n, 'LineWidth',1.5);
grid on; 
xlabel('x'); ylabel('PDF');
title('Double Sided Inverse Gaussian Noise, \mu = 1.5, \lambda = 1')
%}

% generate binary data signal:
N = 1e3; % number of sample points
%rN = randi([-1 1], 1, N); % random integers between -1 and 1
rN = rand(1, N); % random integers between 0 and 1
data = round(rN); clear rN;
% data(data == 0) = -1; % Causes problems with BPSK

% carrier wave parameters:
Fc = 2; % carrier frequency
Fs = 128; % sampling Frequency
cycles = 1; 
Tb = cycles/Fc; % bandwidth of carrier signal
t = 0:1/Fs:(cycles-1/Fs); % time vector used for carrier wave
xC = cos(2*pi*Fc*t); % BPSK modulator equation originally: xC = cos(2*pi*t);
A = 1; % amplitude

%{
% Using awgn for noise. Delete when ready.
%{
st takes size of bits
bits takes size of bt
%}
bt = [];
carrierSignal = [];

i=1;
while (i<N)
    if (data(i))
        bt = [bt ones(1, length(xC))];
    else
        bt = [bt zeros(1, length(xC))];
    end
    carrierSignal = [carrierSignal A*xC];
    i=i+1;
end

bits = 2*(bt - 0.5);
st = carrierSignal.*bits;
nt = random('InverseGaussian',1.5,1,[1 length(st)]); % inverse gaussian noise
rt = awgn(st,10);
%}

bt = [];
carrierSignal = [];

i=1;
while (i<N)
    if (data(i))
        bt = [bt ones(1, length(xC))];
    else
        bt = [bt zeros(1, length(xC))];
    end
    carrierSignal = [carrierSignal A*xC];
    i=i+1;
end
clear i;

bits = 2*(bt - 0.5);
st = carrierSignal.*bits; % BPSK modulation

% add noise to binary bpsk modulated signal
nt = random('InverseGaussian',1.5,1,[1 length(st)]); % inverse gaussian noise
rt = [];
for i = 1:length(st)
    rt(i) = st(i) + 0.1*nt(i);
end

%% 1. Plot Data, BPSK, and BPSK + Noise
figure('Name', 'Binary Data Waveform');

subplot(311);
stairs(bits, 'LineWidth', 1.5);
grid on; ylim([-1.25 1.25]); xlim([0 500]); 
xlabel('time'); 
title('Binary Data')

% figure('Name', 'BPSK');
subplot(312);
plot(st, '*'); 
grid on; ylim([-1.25 1.25]); xlim([0 500]); 
xlabel('time'); 
title('BPSK Modulated')

% figure('Name', 'BPSK Signal with added noise');
subplot(313);
plot(rt,'*'); 
grid on; ylim([-1.25 1.25]); xlim([0 500]); 
xlabel('time'); 
title('BPSK + Noise')

%% 1. Coherent Detector/Demod
% rt = received signal
x = real(rt);
x = conv(x, ones(1, 2));
x = x(2:2:end);
ak = (x>0);

z1 = pskdemod(st,2); % Theoretical
x2 = pskdemod(rt,2); % With noise

%% 1. BER parameters
k = log2(2); % Bits per symbol (log2(Modulation Order))
EbNo = (5:20)'; % Eb/No values (dB) Along x-axis
symbolRate = 100; % number of symbols per frame

berEst = zeros(size(EbNo));
%{
for n = 1:length(EbNo)
    snrdB = EbNo(n) + 10*log10(k);
    numErrors = 0;
    numBits = 0;
    
    while numErrors < 200 && numBits < 1e7
        
        % Calculate the number of bit errors
        nErrors1 = biterr(st,z1); % Theoretical
        % nErrors2 = biterr(rt,z2); % With noise
        
        % Increment the error and bit counters
        numErors = numErors + nErrors1;
        numBits = numBits + symbolRate*k;
        
    end
    
    berEst(n) = numError/numBits;    
end
%}

berTheory = berawgn(EbNo,'psk',2,'nondiff');
figure;
% semilogy(EbNo, berEst, '*')
semilogy(EbNo, berTheory)


%{
% BER Parameters (Delete?)
Eb = (A^2*Tb)/2;
Eb_N0_db = 0:2:14;
Eb_N0 = 10.^(Eb_N0_db/10);
nVar = (Eb)./Eb_N0;
%}

%% 2.
%clear; close all; clc;
%{
You are given the following 8-bits. Obtain the pi/4-DQPSK signal.
1 -1 1 -1 -1 1 1 1 

Do the phase shifts match the theoretical ones? 

Now generate a larger set and obtain the power spectral density.
%}

%% 2. Generate pi/4- DQPSK signal
bits = [1 -1 1 -1 -1 1 1 1];

figure;
stairs(bits, 'LineWidth', 1.5);
grid on; ylim([-1.25 1.25]);
title('Given 8-bits')





