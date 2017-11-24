function [top] = nms(boxes, overlap)
% change format from [x, y, w, h] to [x1, y1, x2, y2]
boxes(:,3) = boxes(:,1) + boxes(:,3);
boxes(:,4) = boxes(:,2) + boxes(:,4);

top = nms_fast(boxes, overlap); 
top(:,3) = top(:,3) - top(:,1);
top(:,4) = top(:,4) - top(:,2);
end