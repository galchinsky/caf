function [f, grad] = fit(X, y1, x1, n, m, c, img, net, clsidx)
fprintf('.');
X = single(reshape(X, n, m, c));
img(y1, x1, :) = X;

res = net.forward({img});
res = res{1};

%[maxval maxidx] = max(res(:));
%maxidx
maxval = res(clsidx);
maxidx = clsidx;

rest = 0*sum(res(1:end ~= maxidx));
f = -(maxval - rest);

ideal_res = res(:)*0;
ideal_res(maxidx) = 1;
delta = ideal_res - res;
delta = -single(delta);


grad = net.backward({delta});
grad = grad{1}(y1, x1, :);

f = double(f(:));
grad = double(grad(:));

end
