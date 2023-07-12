%reset function
function[InitialObservation, LoggedSignals] = PendulumResetFcn()

meanangle = 1; 
%set initial theta
x = 2*(rand - 0.5); %select a random value between -1 and 1
x1 = x*meanangle*0.1;     %multiply random value by 10% of pi/2
theta0 = meanangle + x1; %add random value to pi/2
CosTheta0 = cos(theta0);
SinTheta0 = sin(theta0);
thetadot0 = 0;
%return initial state variables as signals
LoggedSignals.State = [CosTheta0;SinTheta0;thetadot0];
InitialObservation = LoggedSignals.State;
end