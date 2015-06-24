addpath /home/dmitry/caffe/matlab
addpath /home/dmitry/caf

model = '/home/dmitry/caffe/models/bvlc_reference_caffenet/deploy.prototxt';
weights = '/home/dmitry/caffe/models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel';

%model = '/home/dmitry/caffe/models/bvlc_googlenet/deploy.prototxt';
%weights = '/home/dmitry/caffe/models/bvlc_googlenet/bvlc_googlenet.caffemodel';


caffe.set_mode_gpu();

net = caffe.Net(model, weights, 'test'); 
net.blobs('data').reshape([227 227 3 1]);

%im_data = caffe.io.load_image('/home/dmitry/caffe/examples/images/cat.jpg');
im_data = caffe.io.load_image('/home/dmitry/cat.jpg');
im_data = double(im_data);

im = imresize(im_data, [227 227]);
imm = im - mean(im(:));

[n m k] = size(imm);

blocks_y = 1:round(n/120):n;
blocks_x = 1:round(m/120):m;

%blocks_y = 1:n;
%blocks_x = 1:m;

blocks_y = [1 n];
blocks_x = [1 m];

options = optimset('GradObj', 'on' ,'Display', 'iter', 'HessUpdate', 'bfgs', 'GoalsExactAchieve',0, 'TolFun',1e-9); 

imm2 = imm;
for c_y = 2:length(blocks_y)
    for c_x = 2:length(blocks_x)
        y = blocks_y(c_y-1):blocks_y(c_y)
        x = blocks_x(c_x-1):blocks_x(c_x)
        im_part = imm2(y, x, :);
        [n1, m1, c1] = size(im_part);
        l = length(im_part(:));
        %X = ga(@(X) fit(X, y, x, n1, m1, c1, imm2, net), l, [], [], [], [], -20*ones(1,l), 20*ones(1,l), [], gaoptimset('display','iter'));
        CLSIDX = 286;
        X = fminlbfgs(@(X) fit(X, y, x, n1, m1, c1, imm2, net, CLSIDX), im_part(:), options);
        %for i = 1:100
        %    X = X - mean(X(:));
        %    X = X / max(X(:));
        %    X = fminlbfgs(@(X) fit(X, y, x, n1, m1, c1, imm2, net, CLSIDX), X + random('unif',-20,20,length(X),1), options);
        %end
        img = single(reshape(X, n1, m1, c1));
        imm2(y, x, :) = img;
        save a
    end
end

show_blob(imm2)