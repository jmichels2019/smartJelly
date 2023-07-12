%step function
function[NextObs, Reward, IsDone, LoggedSignals] = PendulumStepFcn(Action,...
    LoggedSignals)

Gravity = 9.81;
PendulumMass = 1;
RodLength = 1;
%RodInertia = 0;
%DampingRatio = 0;

Tstep = 0.05;
Tsim = 20;

%MaxAngle = 2*pi;

Torque = Action;


%Unpack state vector from logged signals
State = LoggedSignals.State;
%theta = wrapTo2Pi(theta) == State(1);
CosTheta = State(1)
SinTheta = State(2)
thetadot = State(3)
theta = acos(CosTheta)

%Equation of motion
%thetadot = diff(theta);
thetadotdot = -Gravity*SinTheta/RodLength

%Euler Integration
thetadot1 = Tstep*thetadotdot + thetadot  %find new thetadot from thetadotdot
theta1 = Tstep*thetadot + theta %find new theta from thetadot
CosTheta1 = cos(theta1)
SinTheta1 = sin(theta1)

%transform state to observation
NextObs = LoggedSignals.State;
CosTheta1 = NextObs(1);
SinTheta1 = NextObs(2);
thetadot1 = NextObs(3);

%Calculate Reward
Reward = -((pi - theta1) + 0.1*thetadot1 + 0.001*(1 - CosTheta));
%check terminal condition
IsDone = false;

% get reward
if ~IsDone
    Reward = Reward;
end

end

