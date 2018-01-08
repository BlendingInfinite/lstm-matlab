% The MIT License (MIT)
% Copyright (c) 2018 Moritz Nakatenus
% 
% Permission is hereby granted, free of charge, to any person obtaining a 
% copy of this software and associated documentation files (the "Software"), 
% to deal in the Software without restriction, including without limitation 
% the rights to use, copy, modify, merge, publish, distribute, sublicense, 
% and/or sell copies of the Software, and to permit persons to whom the 
% Software is furnished to do so, subject to the following conditions:
% 
% The above copyright notice and this permission notice shall be included
% in all copies or substantial portions of the Software.
% 
% THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS 
% OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
% FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL 
% THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
% LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING 
% FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER 
% DEALINGS IN THE SOFTWARE.

classdef lstm_network < handle
    
    properties
        layers
        timesteps
        inputDim, hiddenDim, outputDim
        
        input
        outputs
        
        eta = 0.8 % momentum term for Momentum optimizer
        alpha = 0.02 % learning rate
        gamma = 0.9 % decaying rate for the RMSprop optimizer
        epsilon = 10e-8; % for the RMSprop optimizer to prevent numerical problems
        
        optimizer
        
        % parameters (weights, bias as struct)
        weights
    end
    
    methods
        
        function [obj] = lstm_network(timesteps, inputDim, hiddenDim, optimizer)
            
            obj.optimizer = optimizer;
            
            obj.timesteps = timesteps;
            
            % input and output dimensions must be equal
            % to train one mapping matrix for both
            obj.inputDim = inputDim;
            obj.outputDim = obj.inputDim;
            obj.hiddenDim = hiddenDim;
            
            obj.outputs = zeros(obj.outputDim,timesteps);
            
            % initialize weights
            % choose intervall [-a,a]
            a = 1/sqrt(obj.hiddenDim);
            
            obj.weights.W_i = -a + 2*a * rand(obj.hiddenDim,obj.inputDim+2*obj.hiddenDim);
            obj.weights.W_f = -a + 2*a * rand(obj.hiddenDim,obj.inputDim+2*obj.hiddenDim);
            obj.weights.W_c = -a + 2*a * rand(obj.hiddenDim,obj.inputDim+obj.hiddenDim);
            obj.weights.W_o = -a + 2*a * rand(obj.hiddenDim,obj.inputDim+2*obj.hiddenDim);
            
            obj.weights.b_i = -a + 2*a * rand(obj.hiddenDim, 1);
            obj.weights.b_f = -a + 2*a * rand(obj.hiddenDim, 1);
            obj.weights.b_c = -a + 2*a * rand(obj.hiddenDim, 1);
            obj.weights.b_o = -a + 2*a * rand(obj.hiddenDim, 1);
            
            obj.weights.V = -a + 2*a * rand(obj.outputDim,obj.hiddenDim); % softmax layer
            
            for i=1:timesteps
                obj.layers{i} = lstm_layer(obj.outputDim,obj.hiddenDim);
            end
            
            obj.updateLayers();
        end
        
        function [lossCE] = crossEntropyCost(obj, y, t)  
            % crossEntropyCost Softmax cross-entropy loss function
            %   computes the mean cross-entropy cost for all timesteps
            % crossEntropyCost Arguments:
            %   y - output of the softmax layer
            %   t - target value for the softmax layer output
            lossCE = - sum(t .* log(y), 1);
            lossCE = mean(lossCE); % mean over all timesteps
        end
        
        function [lossMSE] = mseCost(obj, y, t)  
            % mseCost Mean-Squared-Error loss function
            %   computes the mean-squared-error cost for all timesteps
            % mseCost Arguments:
            %   y - output of the unsaturated layer
            %   t - target value for the unsaturated layer output
            lossMSE = mean((y-t).^2, 1);
            lossMSE = mean(lossMSE); % mean over all timesteps
        end
       
        function [cost, y_pred] = TruncatedBPTT(obj, y_t, X, k1, k2)
            % TruncatedBPTT Truncated Backpropagation Through Time algorithm
            % BPTT Arguments:
            %   y_t - softmax output targets for all timesteps
            %   X   - input data
            %   k1  - run BPTT every k1 timesteps
            %   k2  - in each BPTT step go k2 timesteps back in time
            
            if(strcmp(obj.optimizer,'RMSProp'))
                % expectation values of previous squared gradients
                expSqGrad.hidden.dE_dWi = zeros(obj.hiddenDim, obj.inputDim+2*obj.hiddenDim);
                expSqGrad.hidden.dE_dWf = zeros(obj.hiddenDim,obj.inputDim+2*obj.hiddenDim);
                expSqGrad.hidden.dE_dWc = zeros(obj.hiddenDim,obj.inputDim+obj.hiddenDim);
                expSqGrad.hidden.dE_dWo = zeros(obj.hiddenDim,obj.inputDim+2*obj.hiddenDim);
                expSqGrad.output        = zeros(obj.outputDim,obj.hiddenDim);
                expSqGrad.hidden.dE_dbi = zeros(obj.hiddenDim, 1);
                expSqGrad.hidden.dE_dbf = zeros(obj.hiddenDim, 1);
                expSqGrad.hidden.dE_dbc = zeros(obj.hiddenDim, 1);
                expSqGrad.hidden.dE_dbo = zeros(obj.hiddenDim, 1);        
            elseif(strcmp(obj.optimizer,'Momentum'))
                updates.hidden.dE_dWi = zeros(obj.hiddenDim, obj.inputDim+2*obj.hiddenDim);
                updates.hidden.dE_dWf = zeros(obj.hiddenDim,obj.inputDim+2*obj.hiddenDim);
                updates.hidden.dE_dWc = zeros(obj.hiddenDim,obj.inputDim+obj.hiddenDim);
                updates.hidden.dE_dWo = zeros(obj.hiddenDim,obj.inputDim+2*obj.hiddenDim);
                updates.output        = zeros(obj.outputDim,obj.hiddenDim);

                updates.hidden.dE_dbi = zeros(obj.hiddenDim, 1);
                updates.hidden.dE_dbf = zeros(obj.hiddenDim, 1);
                updates.hidden.dE_dbc = zeros(obj.hiddenDim, 1);
                updates.hidden.dE_dbo = zeros(obj.hiddenDim, 1);
            end
            % do forward propagation to get activations for computing the
            % gradients
            forwardPropagation(obj, X);
            
            % compute gradients and update
            for t=k2:k1:obj.timesteps
                gradients = obj.layers{t}.computeGradients(y_t(:,t),{obj.layers{(t-k2+1):t}});
                
                if(strcmp(obj.optimizer,'RMSProp'))
                    expSqGrad = RMSpropOptimizer(obj, expSqGrad, gradients);
                elseif(strcmp(obj.optimizer,'Momentum'))
                    updates = MomentumOptimizer(obj, updates, gradients);
                end
            end
            
            % evaluate network
            updateLayers(obj); % assign updated weights to all layers
            forwardPropagation(obj, X);
            %cost = crossEntropyCost(obj,obj.softmaxOutputs(:,1:obj.timesteps), y_t(:,1:obj.timesteps));
            %sprintf('Cross-Entropy Cost: %d', cost)
            cost = mseCost(obj,obj.outputs(:,1:obj.timesteps), y_t(:,1:obj.timesteps));
            sprintf('MSE: %d', cost)
            
            y_pred = obj.outputs;
        end
        
        function [cost, y_pred] = BPTT(obj, y_t, X)
            % BPTT Backpropagation Through Time algorithm
            % BPTT Arguments:
            %   y_t - softmax output targets for all timesteps
            %   X   - input data
            
            if(strcmp(obj.optimizer,'RMSProp'))
                % expectation values of previous squared gradients
                expSqGrad.hidden.dE_dWi = zeros(obj.hiddenDim, obj.inputDim+2*obj.hiddenDim);
                expSqGrad.hidden.dE_dWf = zeros(obj.hiddenDim,obj.inputDim+2*obj.hiddenDim);
                expSqGrad.hidden.dE_dWc = zeros(obj.hiddenDim,obj.inputDim+obj.hiddenDim);
                expSqGrad.hidden.dE_dWo = zeros(obj.hiddenDim,obj.inputDim+2*obj.hiddenDim);
                expSqGrad.output        = zeros(obj.outputDim,obj.hiddenDim);
                expSqGrad.hidden.dE_dbi = zeros(obj.hiddenDim, 1);
                expSqGrad.hidden.dE_dbf = zeros(obj.hiddenDim, 1);
                expSqGrad.hidden.dE_dbc = zeros(obj.hiddenDim, 1);
                expSqGrad.hidden.dE_dbo = zeros(obj.hiddenDim, 1);        
            elseif(strcmp(obj.optimizer,'Momentum'))
                updates.hidden.dE_dWi = zeros(obj.hiddenDim, obj.inputDim+2*obj.hiddenDim);
                updates.hidden.dE_dWf = zeros(obj.hiddenDim,obj.inputDim+2*obj.hiddenDim);
                updates.hidden.dE_dWc = zeros(obj.hiddenDim,obj.inputDim+obj.hiddenDim);
                updates.hidden.dE_dWo = zeros(obj.hiddenDim,obj.inputDim+2*obj.hiddenDim);
                updates.output        = zeros(obj.outputDim,obj.hiddenDim);

                updates.hidden.dE_dbi = zeros(obj.hiddenDim, 1);
                updates.hidden.dE_dbf = zeros(obj.hiddenDim, 1);
                updates.hidden.dE_dbc = zeros(obj.hiddenDim, 1);
                updates.hidden.dE_dbo = zeros(obj.hiddenDim, 1);
            end
            % do forward propagation to get activations for computing the
            % gradients
            forwardPropagation(obj, X);
            
            % compute gradients and update
            for t=1:obj.timesteps
                gradients = obj.layers{t}.computeGradients(y_t(:,t),{obj.layers{1:t}});
                
                if(strcmp(obj.optimizer,'RMSProp'))
                    expSqGrad = RMSpropOptimizer(obj, expSqGrad, gradients);
                elseif(strcmp(obj.optimizer,'Momentum'))
                    updates = MomentumOptimizer(obj, updates, gradients);
                end
            end
            
            % evaluate network
            updateLayers(obj); % assign updated weights to all layers
            forwardPropagation(obj, X);
            %cost = crossEntropyCost(obj,obj.softmaxOutputs(:,1:obj.timesteps), y_t(:,1:obj.timesteps));
            %sprintf('Cross-Entropy Cost: %d', cost)
            cost = mseCost(obj,obj.outputs(:,1:obj.timesteps), y_t(:,1:obj.timesteps));
            sprintf('MSE: %d', cost)
            
            y_pred = obj.outputs;
        end
        
        function [expSqGrad] = RMSpropOptimizer(obj, expSqGrad, gradients)
            % RMSpropOptimizer Optimizer to train the network
            % RMSpropOptimizer Arguments:
            %   expSqGrad - expectation value of previous squared gradients
            %   gradients - gradients of current timestep
            expSqGrad.hidden.dE_dWi = obj.gamma .* expSqGrad.hidden.dE_dWi + ...
                                      (1-obj.gamma) .* gradients.hidden.dE_dWi.^2;
            expSqGrad.hidden.dE_dWf = obj.gamma .* expSqGrad.hidden.dE_dWf + ...
                                      (1-obj.gamma) .* gradients.hidden.dE_dWf.^2;
            expSqGrad.hidden.dE_dWc = obj.gamma .* expSqGrad.hidden.dE_dWc + ...
                                      (1-obj.gamma) .* gradients.hidden.dE_dWc.^2;
            expSqGrad.hidden.dE_dWo = obj.gamma .* expSqGrad.hidden.dE_dWo + ...
                                      (1-obj.gamma) .* gradients.hidden.dE_dWo.^2;
            expSqGrad.output = obj.gamma .* expSqGrad.output + (1-obj.gamma) .* gradients.output.^2;
            
            expSqGrad.hidden.dE_dbi = obj.gamma .* expSqGrad.hidden.dE_dbi + ...
                                      (1-obj.gamma) .* gradients.hidden.dE_dbi.^2;
            expSqGrad.hidden.dE_dbf = obj.gamma .* expSqGrad.hidden.dE_dbf + ...
                                      (1-obj.gamma) .* gradients.hidden.dE_dbf.^2;
            expSqGrad.hidden.dE_dbc = obj.gamma .* expSqGrad.hidden.dE_dbc + ...
                                      (1-obj.gamma) .* gradients.hidden.dE_dbc.^2;
            expSqGrad.hidden.dE_dbo = obj.gamma .* expSqGrad.hidden.dE_dbo + ...
                                      (1-obj.gamma) .* gradients.hidden.dE_dbo.^2;
            
            obj.weights.W_i = obj.weights.W_i - obj.alpha ./ ...
                                      (sqrt(expSqGrad.hidden.dE_dWi + obj.epsilon)) .* gradients.hidden.dE_dWi;
            obj.weights.W_f = obj.weights.W_f - obj.alpha ./ ...
                                      (sqrt(expSqGrad.hidden.dE_dWf + obj.epsilon)) .* gradients.hidden.dE_dWf;
            obj.weights.W_c = obj.weights.W_c - obj.alpha ./ ...
                                      (sqrt(expSqGrad.hidden.dE_dWc + obj.epsilon)) .* gradients.hidden.dE_dWc;
            obj.weights.W_o = obj.weights.W_o - obj.alpha ./ ...
                                      (sqrt(expSqGrad.hidden.dE_dWo + obj.epsilon)) .* gradients.hidden.dE_dWo;
                                  
            obj.weights.b_i = obj.weights.b_i - obj.alpha ./ ...
                                      (sqrt(expSqGrad.hidden.dE_dbi + obj.epsilon)) .* gradients.hidden.dE_dbi;
            obj.weights.b_f = obj.weights.b_f - obj.alpha ./ ...
                                      (sqrt(expSqGrad.hidden.dE_dbf + obj.epsilon)) .* gradients.hidden.dE_dbf;
            obj.weights.b_c = obj.weights.b_c - obj.alpha ./ ...
                                      (sqrt(expSqGrad.hidden.dE_dbc + obj.epsilon)) .* gradients.hidden.dE_dbc;
            obj.weights.b_o = obj.weights.b_o - obj.alpha ./ ...
                                      (sqrt(expSqGrad.hidden.dE_dbo + obj.epsilon)) .* gradients.hidden.dE_dbo;                                  
                                  
            obj.weights.V = obj.weights.V - obj.alpha ./ (sqrt(expSqGrad.output + obj.epsilon)) .* gradients.output;
        end
        
        function [updates] = MomentumOptimizer(obj, lastUpdates, gradients)
            updates.hidden.dE_dWi = obj.eta * lastUpdates.hidden.dE_dWi ...
                                    + obj.alpha * gradients.hidden.dE_dWi;
            updates.hidden.dE_dWf = obj.eta * lastUpdates.hidden.dE_dWf ...
                                    + obj.alpha * gradients.hidden.dE_dWf;
            updates.hidden.dE_dWc = obj.eta * lastUpdates.hidden.dE_dWc ...
                                    + obj.alpha * gradients.hidden.dE_dWc;
            updates.hidden.dE_dWo = obj.eta * lastUpdates.hidden.dE_dWo ...
                                    + obj.alpha * gradients.hidden.dE_dWo;
            updates.output = obj.eta * lastUpdates.output ...
                                    + obj.alpha * gradients.output;
                                
            updates.hidden.dE_dbi = obj.eta * lastUpdates.hidden.dE_dbi ...
                                    + obj.alpha * gradients.hidden.dE_dbi;
            updates.hidden.dE_dbf = obj.eta * lastUpdates.hidden.dE_dbf ...
                                    + obj.alpha * gradients.hidden.dE_dbf;
            updates.hidden.dE_dbc = obj.eta * lastUpdates.hidden.dE_dbc ...
                                    + obj.alpha * gradients.hidden.dE_dbc;
            updates.hidden.dE_dbo = obj.eta * lastUpdates.hidden.dE_dbo ...
                                    + obj.alpha * gradients.hidden.dE_dbo;
                                
            obj.weights.W_i = obj.weights.W_i - updates.hidden.dE_dWi;
            obj.weights.W_f = obj.weights.W_f - updates.hidden.dE_dWf;
            obj.weights.W_c = obj.weights.W_c - updates.hidden.dE_dWc;
            obj.weights.W_o = obj.weights.W_o - updates.hidden.dE_dWo;
            
            obj.weights.b_i = obj.weights.b_i - updates.hidden.dE_dbi;
            obj.weights.b_f = obj.weights.b_f - updates.hidden.dE_dbf;
            obj.weights.b_c = obj.weights.b_c - updates.hidden.dE_dbc;
            obj.weights.b_o = obj.weights.b_o - updates.hidden.dE_dbo;
            
            obj.weights.V = obj.weights.V - updates.output;            
        end
        
        % x: mxn -> m input_vector for timestep n
        function [outputs] = forwardPropagation(obj, X)
            
            h = zeros(obj.hiddenDim,1);
            C = zeros(obj.hiddenDim,1);
            
            for t=1:obj.timesteps
                
                obj.layers{t}.last_h = h;
                obj.layers{t}.last_C = C;
                
                [y,h]=obj.layers{t}.activateLayer(X(:,t));
                
                obj.outputs(:,t) = y;
            end
            
            outputs = obj.outputs;
        end
        
        function updateLayers(obj)
            % updateLayers update layers with the new trained weight matrices

            for t=1:obj.timesteps
                
                obj.layers{t}.weights.W_i = obj.weights.W_i;
                obj.layers{t}.weights.W_f = obj.weights.W_f;
                obj.layers{t}.weights.W_c = obj.weights.W_c;
                obj.layers{t}.weights.W_o = obj.weights.W_o;
                
                obj.layers{t}.weights.b_i = obj.weights.b_i;
                obj.layers{t}.weights.b_f = obj.weights.b_f;
                obj.layers{t}.weights.b_c = obj.weights.b_c;
                obj.layers{t}.weights.b_o = obj.weights.b_o;
                
                obj.layers{t}.weights.V = obj.weights.V;
            end
        end
        
        function [Y] = getOutputs(obj)
            Y = obj.outputs;
        end
        
    end
    
end