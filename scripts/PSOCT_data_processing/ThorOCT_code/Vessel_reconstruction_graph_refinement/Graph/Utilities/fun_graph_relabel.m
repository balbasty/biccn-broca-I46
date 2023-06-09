function input_graph = fun_graph_relabel(input_graph)
% fun_graph_relabel computes the labels for the links and nodes, update all
% graph structure after graph pruning. 
% Input: 
%   input_graph: structure created by fun_skeleton_to_graph. Required
%   field: 
%         num.skeleton_voxel need to be updated.
%         node.cc_ind
%         node.connected_link_label
%         link.cc_ind
%         link.connected_node_label
%         endpoint.pos_ind
% Output: 
%   input_graph: updated strucutre, with all the node/link label
%   continupus. 
% Implemented by Xiang Ji on 11/11/2018
% Modified by Xiang Ji on 02/17/2019:
% 1. No longer require endpoint.link_label. This field is generated by
% link.map_ind_2_label now. The pos_ind is 0 if this endpoint has been
% removed. 
remain_node_label = find(~cellfun(@isempty, input_graph.node.cc_ind));
% Shift 1 for the endpoint (0) 
node_label_map = zeros(numel(input_graph.node.cc_ind) + 1,1);
input_graph.node.num_cc = numel(remain_node_label);
node_label_map(remain_node_label+1) = 1 : input_graph.node.num_cc;

remain_link_list_idx = find(~cellfun(@isempty, input_graph.link.cc_ind));
link_label_map = zeros(numel(input_graph.link.cc_ind), 1);
input_graph.link.num_cc = numel(remain_link_list_idx);
link_label_map(remain_link_list_idx) = 1 : input_graph.link.num_cc;

input_graph.node.cc_ind = input_graph.node.cc_ind(remain_node_label);
tmp_connected_link_label = cell(input_graph.node.num_cc,1);
for tmp_idx = 1 : input_graph.node.num_cc
    tmp_ind_list = link_label_map(input_graph.node.connected_link_label{remain_node_label(tmp_idx)});
    tmp_connected_link_label{tmp_idx} = tmp_ind_list(tmp_ind_list>0);
end
input_graph.node.connected_link_label = tmp_connected_link_label;

%% Links 
input_graph.link.cc_ind = input_graph.link.cc_ind(remain_link_list_idx);
input_graph.link.connected_node_label = node_label_map(input_graph.link.connected_node_label(remain_link_list_idx,:)+1);
% Link derivative fields
input_graph.link.pos_ind = cat(1, input_graph.link.cc_ind{:});
input_graph.link.num_voxel = numel(input_graph.link.pos_ind);
input_graph.link.num_voxel_per_cc = cellfun(@numel, input_graph.link.cc_ind);
input_graph.link.num_node = sum(input_graph.link.connected_node_label>0,2);
if input_graph.link.num_voxel_per_cc > 0
    input_graph.link.label = repelem(1:input_graph.link.num_cc, input_graph.link.num_voxel_per_cc)';
else
    input_graph.link.label = [];
end
input_graph.link.map_ind_2_label = sparse(input_graph.link.pos_ind, ...
    ones(input_graph.link.num_voxel,1), ...
    input_graph.link.label, ...
    input_graph.num.block_voxel,1);
%% Nodes - derivative fields
input_graph.node.pos_ind = cat(1, input_graph.node.cc_ind{:});
input_graph.node.num_voxel = numel(input_graph.node.pos_ind);
input_graph.node.num_voxel_per_cc = cellfun(@numel, input_graph.node.cc_ind);
input_graph.node.num_link = cellfun(@numel, input_graph.node.connected_link_label);
if input_graph.node.num_cc
    input_graph.node.label = repelem(1:input_graph.node.num_cc, input_graph.node.num_voxel_per_cc)';
else
    input_graph.node.label = [];
end
input_graph.node.map_ind_2_label = sparse(input_graph.node.pos_ind, ...
    ones(input_graph.node.num_voxel,1), ...
    input_graph.node.label, ...
    input_graph.num.block_voxel,1);

%% Endpoints
% Find all the nonzero endpoint index
input_graph.endpoint.pos_ind = input_graph.endpoint.pos_ind(input_graph.endpoint.pos_ind > 0);
input_graph.endpoint.link_label =  full(input_graph.link.map_ind_2_label(input_graph.endpoint.pos_ind));
assert(all(input_graph.endpoint.link_label > 0), 'Exist endpoint voxel not part of the link voxel');
input_graph.endpoint.num_voxel = numel(input_graph.endpoint.pos_ind);
input_graph.endpoint.map_ind_2_label = sparse(input_graph.endpoint.pos_ind, ones(input_graph.endpoint.num_voxel, 1),...
    1 : input_graph.endpoint.num_voxel, input_graph.num.block_voxel, 1);
%% 
input_graph.num.skeleton_voxel = input_graph.link.num_voxel + input_graph.node.num_voxel + input_graph.isopoint.num_voxel;
if isfield(input_graph, 'tmp_link_ind_old_to_new')
    input_graph = rmfield(input_graph, 'tmp_link_ind_old_to_new');
end
if isfield(input_graph.link, 'features')
    input_graph.link = rmfield(input_graph.link, 'features');
end
if isfield(input_graph.node, 'features')
    input_graph.node = rmfield(input_graph.node, 'features');
end

end