clear all;
close all;

%define and open our model
mdl = "rlSimplePendulumModel";
open_system(mdl)

%observationinfo
ObservationInfo = rlNumericSpec([3 1]);
ObservationInfo_Name = "Pendulum States";
ObservationInfo_Description = 'theta, CosTheta, SinTheta, thetadot';
%actioninfo
ActionInfo = rlNumericSpec([1 1],'LowerLimit',[-2],'UpperLimit',[2]);
ActionInfo_Name = "Pendulum Action";

%define our environment
%env = rlFunctionEnv(ObservationInfo,ActionInfo,"PendulumStepFcn",...
%    "PendulumResetFcn")
env = rlPredefinedEnv("SimplePendulumModel-Continuous")

%get observation from actions and environment
obsInfo = getObservationInfo(env);
actInfo = getActionInfo(env)

%set obervations to be sine of pendulum angle, cos of pendulum angle,
% and derivative of pendulum angle
set_param("rlSimplePendulumModel/create observations", ... 
    "ThetaObservationHandling", "sincos");

env.ResetFcn = @(in)setVariable(in,"theta0",pi,"Workspace",mdl);
%define intial condition of pendulum as hanging down, specify env. reset
%reset function sets model workspace variable theta0 to pi
%specify sim time Tf as 20s and sample time Ts as 0.05s
Tstep = 0.05;
Tsim = 20;
%fix random generator seed for reproducibility
rng(0)

%Now we create DDPG agent
%Define state path
statePath = [
    featureInputLayer(obsInfo.Dimension(1), Name="obsPathInputLayer")
    fullyConnectedLayer(40) %changed from 400 to 40
    reluLayer
    fullyConnectedLayer(30,Name="spOutLayer") %changed from 300 to 30
    ];
%Define action path
actionPath = [
    featureInputLayer(actInfo.Dimension(1), Name="actPathInputLayer")
    fullyConnectedLayer(30,Name = "apOutLayer", BiasLearnRateFactor=0) %changed fro, 300 to 30
    ];
%Define common path
commonPath = [
    additionLayer(2,Name="add")
    reluLayer %rectified linear unit layer
    fullyConnectedLayer(1) %connected layer
    ];
%Create layer graph, add & connect layers
criticNetwork = layerGraph();
criticNetwork = addLayers(criticNetwork,statePath);
criticNetwork = addLayers(criticNetwork,actionPath);
criticNetwork = addLayers(criticNetwork,commonPath);
criticNetwork = connectLayers(criticNetwork,"spOutLayer","add/in1"); %connects layer to statePath
criticNetwork = connectLayers(criticNetwork,"apOutLayer","add/in2"); %connects layer to actionPath

%convert to dlnetwork and display number of weights
criticNetwork = dlnetwork(criticNetwork);
summary(criticNetwork)

%view critic network config
plot(criticNetwork)

%create the critic object using criticNetwork, environment obs and action
%specs, and names of network layers to be connected
critic = rlQValueFunction(criticNetwork, obsInfo, actInfo, ...
    ObservationInputNames="obsPathInputLayer", ...
    ActionInputNames="actPathInputLayer");
%define the actor network as an array of layers. actor takes current
%observation as input and and outputs an action. since the output is always
%between -1 and 1, use scaling layer to scale the range of action
%(actInfo.Upperlimit)
actorNetwork = [
    featureInputLayer(obsInfo.Dimension(1))
    fullyConnectedLayer(40) %changed from 400 to 40
    reluLayer
    fullyConnectedLayer(30) %changed from 300 to 30
    reluLayer
    fullyConnectedLayer(1)
    tanhLayer %establishes hyperbolic tangent layer
    scalingLayer(Scale=max(actInfo.UpperLimit))
    ];
%convert actor network to dlnetwork
actorNetwork = dlnetwork(actorNetwork);
summary(actorNetwork)
%create the actor using actorNet and add observation and action specs
actor = rlContinuousDeterministicActor(actorNetwork,obsInfo,actInfo);
%specify options for critic and actor
criticOpts = rlOptimizerOptions(LearnRate=1e-03,GradientThreshold=1);
actorOpts = rlOptimizerOptions(LearnRate=1e-04,GradientThreshold=1);

%Specify the DDPG agent options 
agentOpts = rlDDPGAgentOptions(SampleTime=Tstep, ...
    CriticOptimizerOptions=criticOpts, ActorOptimizerOptions=actorOpts, ...
    ExperienceBufferLength=1e6, DiscountFactor=0.99, MiniBatchSize=128);
%Use dot notation to modify the agent opts
agentOpts.NoiseOptions.Variance = 0.6;
agentOpts.NoiseOptions.VarianceDecayRate = 1e-5;
%create DDPG agent using specified actor, critic, and options objects
agent = rlDDPGAgent(actor, critic, agentOpts);

%Train Agent
maxepisodes = 5000; %train for at most 5000 episodes, each using Tf and Ts steps
maxsteps = ceil(Tsim/Tstep); %steps defined using Tf and Ts
trainOpts = rlTrainingOptions(MaxEpisodes=maxepisodes, ...
    MaxStepsPerEpisode=maxsteps, ScoreAveragingWindowLength=5, ...
    Verbose=false, Plots="training-progress", ...
    StopTrainingCriteria="AverageReward", StopTrainingValue=740, ...
    SaveAgentCriteria="EpisodeReward", SaveAgentValue=-740);
%we train the agent using the train function. training is long and
%intensive, so for this example we load a pretrained agent by setting
%command doTraining to false. To train on your own, set doTraining to true
doTraining = true;
if doTraining
    trainingStats = train(agent,env,trainOpts); %trains the agent
else
    load("SimulinkPendulumDDPG.mat","agent")
end

%simulate DDPG agent
%to simulate the pendulum environment
simOptions = rlSimulationOptions(MaxSteps=500);
experience = sim(env,agent,simOptions);


