%% Initialization
%  Initialize the world, Q-table, and hyperparameters
clear variables; close all; clf;

world = 1;

gwinit(world);
state = gwstate();

gamma      = 0.998;    % discount rate
epsilon    = 0.95;   % exploration rate
alpha      = 1;    % learning rate
n_episodes = 1000;

actions   = [1, 2, 3, 4];
n_actions = length(actions);

action_probs = 1/4.* [1,1,1,1];

Q_table = randn(state.ysize, state.xsize, n_actions);
gwdraw()

%% Training loop
%  Train the agent using the Q-learning algorithm.

for episode = 1:n_episodes
    
    % update epsilon
    e = epsilon * getepsilon(episode, n_episodes);
    
    % set state to original pos
    gwinit(world);
    state = gwstate();
    
    % loop while state is not in final position
    while ~state.isterminal
        
        % get last state and it's positions
        old_state = state;
        x_old = old_state.pos(2);
        y_old = old_state.pos(1);
        
        
        %choose action to test
        [action, ~] = chooseaction(Q_table, y_old, x_old, actions, action_probs, e);

        %perform action
        gwaction(action);

        % get new state
        state = gwstate();

        % get feedback, if state is invalid (outside borders) feedback is
        % -100
        if (~state.isvalid && world ~= 4)
            feedback = -inf;
        else
            feedback = state.feedback;
        end
        
        % get new state positions
        x_new = state.pos(2);
        y_new = state.pos(1);
        
        % get optimal action for new state
        [~, opt_a] = chooseaction(Q_table, y_new, x_new, actions, action_probs, e);
        
        % update Qtable
        feedback_update = alpha * (feedback + gamma * Q_table(y_new, x_new, opt_a));
        Q_table(y_old, x_old, action) = (1 - alpha) * Q_table(y_old, x_old, action) + feedback_update; 
        
    end
    
    if ~mod(episode, 1000)
        disp(episode)
        disp(e)

        gwdraw();
        ptemp = getpolicy(Q_table);
        gwdrawpolicy(ptemp);
        pause(3);
    end
    
    if state.isterminal
        Q_table(y_new, x_new, :) = 0;
    end
    
end

%% plots
% get policy for the Qtable and draw it
figure(1)
gwdraw();
P = getpolicy(Q_table);
gwdrawpolicy(P);
%%
figure(2)
V = getvalue(Q_table);
imagesc(V);
%% Test loop
%  Test the agent (subjectively) by letting it use the optimal policy
%  to traverse the gridworld. Do not update the Q-table when testing.
%  Also, you should not explore when testing, i.e. epsilon=0; always pick
%  the optimal action.

steps = 0;

figure(3)
clf;
gwdraw()
while true
    gwinit(world);
    state = gwstate();
    while ~state.isterminal



            % get last state and it's positions
            x = state.pos(2);
            y = state.pos(1);


            %choose action to test
            action = P(y, x);

            %perform action
            gwaction(action);

            steps = steps + 1;

            % get new state
            state = gwstate();

            figure(3)
            clf;
            gwdraw()


    end
end
    

