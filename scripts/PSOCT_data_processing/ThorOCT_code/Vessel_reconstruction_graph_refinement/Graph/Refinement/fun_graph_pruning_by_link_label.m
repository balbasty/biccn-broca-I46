function input_graph = fun_graph_pruning_by_link_label(input_graph, link_label_to_remove)
% fun_graph_pruning_by_link_label removes links with free ends accoridng to
% the input link label
% Input: 
%   input_graph: structure, generated by fun_skeleton_to_graph
%   link_label_to_remove: N-by-1 numerical vector. Each element is the
%   label of a link to be removed. 
% Output: 
%   input_graph: graph of the same field as the input
% Note: 
% 1. In this function, link labels are specified, which means the voxel list
% in the link cc should not be changed during pruning. 
% 2. This function differs from fun_graph_delete_internal_links by checking
% if the number of link voxels to be pruned has changed or not. If yes, the
% link will not be deleted. 
%% Remove 
num_link_to_remove = numel(link_label_to_remove);
if num_link_to_remove == 0
%     disp('No link needs to be removed');
    return;
else
%     fprintf('Need to remove %d links\n', num_link_to_remove);
end
tmp_ori_link_length = input_graph.link.num_voxel_per_cc(link_label_to_remove);
input_graph.tmp_link_ind_old_to_new = 0 : input_graph.link.num_cc;
for int_ep_idx = 1 : num_link_to_remove
    link_label = link_label_to_remove(int_ep_idx);  
    % Map the link label to the updated one.
    link_label = input_graph.tmp_link_ind_old_to_new(link_label + 1);
    if tmp_ori_link_length(int_ep_idx) ~= numel(input_graph.link.cc_ind{link_label})
%         fprintf('The length of the link to be removed has changed from %d to %d. Skip this link\n',...
%             tmp_ori_link_length(int_ep_idx), numel(input_graph.link.cc_ind{link_label}));
        continue;
    end
    
    while ~isempty(link_label) && (link_label ~= 0)
        connected_node_label = input_graph.link.connected_node_label(link_label,1);
        % Remove the link:
        [input_graph, updated_node_info] = fun_graph_pruning_remove_single_link(input_graph, link_label, false);
        if isempty(updated_node_info.label)
%             disp('This link does not connect to any node');
            link_label = [];
            continue;
        end
        if updated_node_info.degree == 2
            [input_graph, ~] = fun_graph_pruning_convert_node_to_link(input_graph, connected_node_label);
            link_label = [];
        elseif updated_node_info.degree == 1
            % Can be a loop
%             fprintf('Node %d only connects to single link. Convert it to be an endpoint\n', connected_node_label);
            [input_graph, ~] = fun_graph_pruning_convert_node_to_endpoint(input_graph, connected_node_label);
            link_label = [];
        else
            link_label = [];
        end

    end
end
%% Re-compute the labels
input_graph = fun_graph_relabel(input_graph);
end