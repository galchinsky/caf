q = imm2 - min(imm2(:));
q = q / max(q(:));

q = q(:, :, [3 2 1]);
q = permute(q, [2 1 3]);

q2 = imm - min(imm(:));
q2 = q2 / max(q2(:));
q2 = q2(:, :, [3 2 1]);
q2 = permute(q2, [2 1 3]);
subplot(2,2,1);
imagesc(q2);

subplot(2,2,2);
imagesc(q);