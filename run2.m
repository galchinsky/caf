addpath /home/dmitry/caffe/matlab
addpath /home/dmitry/caf

model = '/home/dmitry/caffe/models/bvlc_reference_caffenet/deploy.prototxt';
weights = '/home/dmitry/caffe/models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel';

%model = '/home/dmitry/caffe/models/bvlc_googlenet/deploy.prototxt';
%weights = '/home/dmitry/caffe/models/bvlc_googlenet/bvlc_googlenet.caffemodel';


caffe.set_mode_gpu();

net = caffe.Net(model, weights, 'test'); 
net.blobs('data').reshape([227 227 3 1]);

im_data = caffe.io.load_image('/home/dmitry/caffe/examples/images/cat.jpg');
%im_data = caffe.io.load_image('/home/dmitry/cat.jpg');
im_data = double(im_data);

im = imresize(im_data, [227 227]);
imm = im - mean(im(:));

[n, m, c] = size(imm);

blocks_y = 1:round(n/120):n;
blocks_x = 1:round(m/120):m;


res = net.forward({imm});

channel = 4;
the_layer = net.blob_vec(channel).get_data();

thresh = quantile(the_layer(:), 0.9);


the_layer(the_layer>thresh) = the_layer(the_layer>thresh) * 1600;

options = optimset('GradObj', 'on' ,'Display', 'iter', 'HessUpdate', 'bfgs', 'GoalsExactAchieve',0, 'TolFun',1e-6, 'MaxIter', 10); 


X = fminlbfgs(@(X) fit2(X, n, m, c, the_layer, channel, net), imm(:), options);

img = single(reshape(X, n, m, c));

show_blob(img)