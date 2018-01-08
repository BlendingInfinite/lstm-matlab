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

classdef lstm_layer < handle
    
    properties
        
        % input/output dimension
        inputDim,hiddenDim,outputDim

        % input, output, input/output/forget gate, state
        h,i,o,f,C,C_tild,y % dh = derivative with respect to whole input
        last_h,last_C
        
        i_arg, f_arg, c_arg, o_arg % arguments of the gates
        i_hArgInd, f_hArgInd, c_hArgInd, o_hArgInd % index-quantity of h in argument of gate
        o_cArgInd % index-quantity of C in argument
        
        % weight-matrices
        weights
 
    end
    
    methods
        
        function obj = lstm_layer(outputDim, hiddenDim) 
            
            %% INITIALIZE LAYER
            % input/output dimension
            obj.outputDim = outputDim;
            obj.inputDim = obj.outputDim;
            obj.hiddenDim = hiddenDim;
            
            % initialize components
            obj.h=zeros(obj.hiddenDim,1);
            obj.i=zeros(obj.hiddenDim,1);
            obj.o=zeros(obj.hiddenDim,1);
            obj.f=zeros(obj.hiddenDim);
            obj.C=zeros(obj.hiddenDim,1);
            obj.C_tild=zeros(obj.hiddenDim,1);
            obj.y=zeros(obj.outputDim,1);
        end
        
        function gradients = computeGradients(obj,y_t,layers)
            % computeGradients
            %	compute gradients for each gate regarding the timerange
            %	from current timestep to the first one
            % computeGradients Arguments:
            %   y_t - target value for softmax output at current timestep
            
            %% INITIALIZE GRADIENTS
            dE_dWi = zeros(obj.hiddenDim,obj.inputDim+2*obj.hiddenDim);
            dE_dWf = zeros(obj.hiddenDim,obj.inputDim+2*obj.hiddenDim);
            dE_dWc = zeros(obj.hiddenDim,obj.inputDim+obj.hiddenDim);
            dE_dWo = zeros(obj.hiddenDim,obj.inputDim+2*obj.hiddenDim);
            
            dE_dbi = zeros(obj.hiddenDim, 1);
            dE_dbf = zeros(obj.hiddenDim, 1);
            dE_dbc = zeros(obj.hiddenDim, 1);
            dE_dbo = zeros(obj.hiddenDim, 1);
            
            %% BACKPROPAGATION THROUGH TIME FOR TIMESTEP t
            delta = obj.y - y_t;
            delta_h = layers{end}.weights.V' * delta; % error with respect to h_T
            
            % gradient for softmax layer
            dE_dV = delta * obj.h';
            
            for t=length(layers):-1:1
                
                if(t == length(layers))
                    gradients = computeGradient(obj, layers{t}, 0, delta_h);
                else
                    gradients = computeGradient(obj, layers{t}, layers{t+1}, delta_h);
                end
                
                % gradients for cell
                dE_dWi = dE_dWi + gradients.hidden.dE_dWi;
                dE_dWf = dE_dWf + gradients.hidden.dE_dWf;
                dE_dWc = dE_dWc + gradients.hidden.dE_dWc;
                dE_dWo = dE_dWo + gradients.hidden.dE_dWo;
                
                dE_dbi = dE_dbi + gradients.hidden.dE_dbi;
                dE_dbf = dE_dbf + gradients.hidden.dE_dbf;
                dE_dbc = dE_dbc + gradients.hidden.dE_dbc;
                dE_dbo = dE_dbo + gradients.hidden.dE_dbo;                
            end
  
            gradients = struct;
            gradients.hidden.dE_dWc = dE_dWc;
            gradients.hidden.dE_dWo = dE_dWo;
            gradients.hidden.dE_dWi = dE_dWi;
            gradients.hidden.dE_dWf = dE_dWf;
            
            gradients.hidden.dE_dbc = dE_dbc;
            gradients.hidden.dE_dbo = dE_dbo;
            gradients.hidden.dE_dbi = dE_dbi;
            gradients.hidden.dE_dbf = dE_dbf;   
            
            gradients.output = dE_dV;
        end
        
        function [gradients, delta_h] = computeGradient(obj, layer, nextLayer, delta_h)
            % computeGradient Compute gradients of the lstm cell regarding one timestep
            % computeGradient Arguments:
            %   layer     - lstm layer of the current timestep
            %   nextLayer - lstm layer of the next timestep
            %   delta_h   - delta_h of the next timestep
            if(nextLayer ~= 0)
                dhNext_dh = nextLayer.weights.W_o(:,obj.o_hArgInd)' * dsigmoid(obj, nextLayer.o) * diag(tanh(nextLayer.C)) ...
                + (nextLayer.weights.W_f(:,obj.f_hArgInd)' * dsigmoid(obj, nextLayer.f) * diag(layer.C) ...
                   + nextLayer.weights.W_i(:,obj.i_hArgInd)' * dsigmoid(obj, nextLayer.i) * diag(nextLayer.C_tild) ...
                   + nextLayer.weights.W_c(:,obj.c_hArgInd)' * dtanh(obj, nextLayer.C_tild) * diag(nextLayer.i)) ...
                * dtanh(obj, nextLayer.C) * diag(nextLayer.o);

                delta_h = dhNext_dh * delta_h;
            end
            
            dE_dC = (layer.weights.W_o(:,obj.o_cArgInd)' * dsigmoid(obj, layer.o) * diag(tanh(layer.C)) ...
                    + dtanh(obj, layer.C) * diag(layer.o)) * delta_h;
                
            delta_i = diag(layer.C_tild) * dE_dC;
            delta_f = diag(layer.last_C) * dE_dC;
            delta_c = diag(layer.i)      * dE_dC;
            delta_o = tanh(layer.C)     .* delta_h;
            
            dE_dWi = (dsigmoidP(obj, layer.i) .* delta_i) * layer.i_arg';
            dE_dWf = (dsigmoidP(obj, layer.f) .* delta_f) * layer.f_arg';
            dE_dWc = (dtanhP(obj, layer.C)    .* delta_c) * layer.c_arg';
            dE_dWo = (dsigmoidP(obj, layer.o) .* delta_o) * layer.o_arg';
 
            dE_dbi = dsigmoid(obj, layer.i) * delta_i;
            dE_dbf = dsigmoid(obj, layer.f) * delta_f;
            dE_dbc = dtanh(obj, layer.C)    * delta_c;
            dE_dbo = dsigmoid(obj, layer.o) * delta_o;
            
            gradients = struct;
            gradients.hidden.dE_dWc = dE_dWc;
            gradients.hidden.dE_dWo = dE_dWo;
            gradients.hidden.dE_dWi = dE_dWi;
            gradients.hidden.dE_dWf = dE_dWf;
            
            gradients.hidden.dE_dbc = dE_dbc;
            gradients.hidden.dE_dbo = dE_dbo;
            gradients.hidden.dE_dbi = dE_dbi;
            gradients.hidden.dE_dbf = dE_dbf;            
        end
        
        function [y,h] = activateLayer(obj,x)
            
            obj.i_arg = [x; obj.last_h; obj.last_C];
            obj.f_arg = [x; obj.last_h; obj.last_C];
            obj.c_arg = [x; obj.last_h];
            obj.o_arg = [x; obj.last_h; obj.C];
            
            % get index quantities of input arguments
            if(isempty(obj.i_hArgInd))
                hLength  = length(obj.last_h);
                xLastInd = find(x==x(end));
                xLastInd = xLastInd(end);
                obj.i_hArgInd = (xLastInd + 1):(xLastInd + hLength);
                obj.f_hArgInd = obj.i_hArgInd;
                obj.c_hArgInd = obj.i_hArgInd;
                obj.o_hArgInd = obj.i_hArgInd;
                
                obj.o_cArgInd = (xLastInd + hLength + 1):(xLastInd + hLength + length(obj.C));
            end
            
            %% FORWARD PROPAGATION STEP
            obj.i      = obj.sigmoid(obj.weights.W_i * obj.i_arg + obj.weights.b_i);
            obj.C_tild = tanh(obj.weights.W_c * obj.c_arg + obj.weights.b_c);
            obj.f      = obj.sigmoid(obj.weights.W_f * obj.f_arg + obj.weights.b_f);
            obj.C      = obj.f.*obj.last_C + obj.i .* obj.C_tild;
            obj.o      = obj.sigmoid(obj.weights.W_o * obj.o_arg + obj.weights.b_o);
            obj.h      = obj.o.*tanh(obj.C);
            
            % compute softmax layer output
            %obj.y = obj.softmax(obj.weights.V * obj.h);
            
            % unsaturated output-layer to compute gradient for MSE.
            % Delta does not change, hence we don't have to change
            % the gradients
            obj.y = obj.weights.V * obj.h;
            
            y = obj.y;
            h = obj.h;
        end
        
        function [argout] = softmax(obj, arg)
            argout = exp(arg)./sum(exp(arg));
        end
        
        function [argout] = sigmoid(obj, arg)
            argout = 1./(1+exp(-arg));
        end
        
        function [argout] = dsigmoid(obj, func)
            % dsigmoid Derivative of sigmoid function
            % dsigmoid Arguments:
            %   func - sigmoid function to derivate
            if(length(func)==1)
                argout = func * (1 - func);
            else
                argout = diag(func .* (1 - func));
            end
        end
        
        function [argout] = dtanh(obj, func)
            % dtanh Derivative of tanh with respect to its input
            % dtanh Arguments:
            %   func - tanh function to derivate
            if(length(func)==1)
                argout = (1 - func^2);
            else
                argout = diag(1 - func.^2);
            end
        end
        
        function [argout] = dsigmoidP(obj, func)
            % dsigmoidP Pointwise derivatives of sigmoid functions
            % dsigmoidP Arguments:
            %   func - sigmoid function to derivate
            argout = func .* (1 - func);
        end
        
        function [argout] = dtanhP(obj, func)
            % dtanh pointwise derivatives of tanh with respect to its input
            % dtanh Arguments:
            %   func - tanh function to derivate
            argout = 1 - func.^2;
        end
    end
    
end
