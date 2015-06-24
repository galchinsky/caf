function show_blob(imm2)

q = imm2 - min(imm2(:));
q = q / max(q(:));

q = q(:, :, [3 2 1]);
q = permute(q, [2 1 3]);
imagesc(q);
