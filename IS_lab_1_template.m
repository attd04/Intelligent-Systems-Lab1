% Classification using perceptron

% create training algorithm w/ single perceptron to classify the given
% objects

% Reading apple images
A1=imread('apple_04.jpg');
A2=imread('apple_05.jpg');
A3=imread('apple_06.jpg');
A4=imread('apple_07.jpg');
A5=imread('apple_11.jpg');
A6=imread('apple_12.jpg');
A7=imread('apple_13.jpg');
A8=imread('apple_17.jpg');
A9=imread('apple_19.jpg');

% Reading pears images
P1=imread('pear_01.jpg');
P2=imread('pear_02.jpg');
P3=imread('pear_03.jpg');
P4=imread('pear_09.jpg');

% Calculate for each image, colour and roundness
% Apples
hsv_value_A1=spalva_color(A1); %color
metric_A1=apvalumas_roundness(A1); %roundness

hsv_value_A2=spalva_color(A2); %color
metric_A2=apvalumas_roundness(A2); %roundness

hsv_value_A3=spalva_color(A3); %color
metric_A3=apvalumas_roundness(A3); %roundness

hsv_value_A4=spalva_color(A4); %color
metric_A4=apvalumas_roundness(A4); %roundness

hsv_value_A5=spalva_color(A5); %color
metric_A5=apvalumas_roundness(A5); %roundness

hsv_value_A6=spalva_color(A6); %color
metric_A6=apvalumas_roundness(A6); %roundness

hsv_value_A7=spalva_color(A7); %color
metric_A7=apvalumas_roundness(A7); %roundness

hsv_value_A8=spalva_color(A8); %color
metric_A8=apvalumas_roundness(A8); %roundness

hsv_value_A9=spalva_color(A9); %color
metric_A9=apvalumas_roundness(A9); %roundness


% pears
hsv_value_P1=spalva_color(P1); %color
metric_P1=apvalumas_roundness(P1); %roundness

hsv_value_P2=spalva_color(P2); %color
metric_P2=apvalumas_roundness(P2); %roundness

hsv_value_P3=spalva_color(P3); %color
metric_P3=apvalumas_roundness(P3); %roundness

hsv_value_P4=spalva_color(P4); %color
metric_P4=apvalumas_roundness(P4); %roundness



%selecting features(color, roundness, 3 apples and 2 pears)
%A1,A2,A3,P1,P2
%building matrix 2x5
x1=[hsv_value_A1 hsv_value_A2 hsv_value_A3 hsv_value_P1 hsv_value_P2];

x2=[metric_A1 metric_A2 metric_A3 metric_P1 metric_P2];

% estimated features are stored in matrix P:
P=[x1;x2];

%desired output vector (correct answers)
T=[1;1;1;-1;-1]; % <- ČIA ANKSČIAU BUVO KLAIDA!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

%% train single perceptron with two inputs and one output

% generate random initial values of w1, w2 and b
% random bc we don't know correct weights
% must discover thru training and correcting errors
w1 = randn(1);
w2 = randn(1);
b = randn(1);

LR = 0.3; % controls how much weights & bias are changing

maxEpochs = 100;

% epoch counter
epoch = 0;

% initialize total error
e = 1;


% training algorithm:
% executes while the total error is not 0
% parameter update: # of epochs
while e ~= 0 && epoch < maxEpochs

    % epoch counter + 1
    epoch = epoch + 1;

    % process training examples
    for k = 1:5

        % calculate weighted sum with randomly generated parameters
        v = x1(k)*w1 + x2(k)*w2 + b;

        % find output for current example
        if v > 0
            y = 1;
        else
            y = -1;
        end

        % find error for current example (desired - actual output)
        error = T(k) - y;

        % update parameters using current inputs & current error
        w1 = w1 + LR*error*x1(k);
        w2 = w2 + LR*error*x2(k);
        b = b + LR*error;
    end


    % test how good are updated parameters (weights) on all examples used for training
    % calculates weighted sums w/ new weights
    for k = 1:5

        % Calculate weighted sum using updated weights
        v_train(k) = x1(k)*w1 + x2(k)*w2 + b;

        % Calculate current perceptron output
        if v_train(k) > 0
            y_train(k) = 1;
        else
            y_train(k) = -1;
        end

        % Calculate error for each example
        w_e(k) = T(k) - y_train(k);

    end

    % calculate the total error for these 5 inputs 
    e = sum(abs(w_e));

end


if e == 0
    fprintf('\nTraining completed successfully\n');
else
    fprintf('\nMaximum number of epochs reached.\n');
end

fprintf('\nFinal parameters:\n');
fprintf('w1 = %.6f\n', w1);
fprintf('w2 = %.6f\n', w2);
fprintf('b = %.6f\n', b);

fprintf('\nFinal classification of training examples:\n');

for k = 1:5
    fprintf('Example %d: Desired = %d, Predicted = %d\n', ...
        k, T(k), y_train(k));
end


% testing:
% images that were NOT used during training
% six apples and two pears

x1_test = [hsv_value_A4 hsv_value_A5 hsv_value_A6 ...
    hsv_value_A7 hsv_value_A8 hsv_value_A9 ...
    hsv_value_P3 hsv_value_P4];

x2_test = [metric_A4 metric_A5 metric_A6 ...
    metric_A7 metric_A8 metric_A9 ...
    metric_P3 metric_P4];

% desired output vector for testing images
T_test = [1; 1; 1; 1; 1; 1; -1; -1];

% initialize 
correct = 0;

% test all eight images using the final trained parameters
fprintf('\nTesting results:\n');


for k = 1:8

    % calculate weighted sum using final trained parameters
    v_test(k) = x1_test(k)*w1 + x2_test(k)*w2 + b;

    % calculate current output of the perceptron
    if v_test(k) > 0
        y_test(k) = 1;
    else
        y_test(k) = -1;
    end

    % compare predicted output with desired output
    if y_test(k) == T_test(k)
        correct = correct + 1;
    end

    fprintf('Test example %d: Desired = %d, Predicted = %d\n', ...
        k, T_test(k), y_test(k));

end

% calculate testing accuracy
accuracy = (correct / 8) * 100;

fprintf('\nCorrect classifications: %d out of 8\n', correct);
fprintf('Testing accuracy: %.2f%%\n', accuracy);