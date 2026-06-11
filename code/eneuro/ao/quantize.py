from __future__ import annotations
from contextlib import contextmanager
from typing import List, Dict, Any, Optional
import numpy as np
import weakref

from .graph import Graph, Node, NodeType

from ..base.core import Tensor, Function, Config
from ..nn.optim import Optimizer
from ..base.functions import to_xp, get_array_module
from ..base import functions as f

"""
class QuantizeManager:
    @staticmethod
    def apply_quantize(graph: Graph) -> Graph:
        # 初始化
        for node in graph.nodes.values():
            node.quantized = False
            node.dequantized = False

        FQ_state_stack = []
        # 按拓扑序遍历所有 Function 节点
        for node in graph.topological_order():
            if node.type != NodeType.FUNCTION: # 排除tensor
                continue

            func_cls = node.true_obj.__class__
            # 遇到FQ/DFQ节点，进行替换并标记后继节点
            if func_cls in (f.FakeQuantize, f.FakeDequantize):
                pre_nodes = graph.get_predecessors(node)
                suc_nodes = graph.get_successors(node)
                for suc_node in suc_nodes:
                    if func_cls == f.FakeQuantize:
                        suc_node.quantized = True
                        suc_node.dequantized = False
                    else:
                        suc_node.dequantized = True
                        suc_node.quantized = False

                if func_cls == f.FakeQuantize:
                    # 替换为真正的Quantize节点
                    new_func = f.Quantize(node.true_obj.scale, node.true_obj.zero_point, node.true_obj.dtype)
                    graph.replace_subgraph([node], pre_nodes, suc_nodes, new_func)
                    FQ_state_stack.append([node.true_obj.scale, node.true_obj.zero_point, node.true_obj.dtype]) # 记录FQ状态以供后续使用
                else:
                    # 替换为真正的Dequantize节点
                    new_func = f.Dequantize(node.true_obj.scale, node.true_obj.zero_point)
                    graph.replace_subgraph([node], pre_nodes, suc_nodes, new_func)
                    FQ_state_stack.pop()

            # 其他Function节点
            else:
                # 获取前继节点状态
                pre_nodes = graph.get_predecessors(node)
                pre_quantized = True
                for pre in pre_nodes:
                    if pre.quantized == False:
                        pre_quantized = False
                        break

                pre_dequantized = True
                for pre in pre_nodes:
                    if pre.dequantized == False:
                        pre_dequantized = False
                        break
                
                # 后继节点
                suc_nodes = graph.get_successors(node)

                if len(FQ_state_stack) > 0: # 只有在当前处于FQ状态（即上游有FQ节点）时才考虑添加Quantize/Dequantize
                    # 可以量化的Function
                    if func_cls in f.QuantizeRegistry.can_quantize:
                        # 若输入不是量化的，则添加Quantize
                        if not pre_quantized:
                            '''
                            pre(Tensor) -> node
                            变为
                            pre(Tensor) -> Quantize -> Tensor -> node 
                            '''
                            scale, zero_point, dtype = FQ_state_stack[-1] 
                            new_func = f.Quantize(scale=scale, zero_point=zero_point, dtype=dtype)
                            # 去除原来的边
                            graph._remove_edges_to_node(node, keep_set=set(pre_nodes))
                            for pre in pre_nodes:
                                # 添加节点
                                quantize_func = f.Quantize(scale=scale, zero_point=zero_point, dtype=dtype)
                                tensor = quantize_func(pre.true_obj)
                                quantize_node = graph.add_node(quantize_func)
                                tensor_node = graph.add_node(weakref.ref(tensor))
                                tensor_node.quantized = True
                                tensor_node.dequantized = False
                                # 添加边
                                graph.add_edge(pre, quantize_node)
                                graph.add_edge(quantize_node, tensor_node)
                                graph.add_edge(tensor_node, node)
                        
                        # 标记后继节点
                        for suc in suc_nodes:
                            suc.quantized = True
                            suc.dequantized = False

                    # 不能量化的Function
                    else:
                        # 若输入不是反量化的，则添加Dequantize
                        if not pre_dequantized:
                            scale, zero_point, dtype = FQ_state_stack[-1] 
                            new_func = f.Dequantize(scale=scale, zero_point=zero_point)
                            '''
                            pre(Tensor) -> node
                            变为
                            pre(Tensor) -> Dequantize -> Tensor -> node 
                            '''
                            # 去除原来的边
                            graph._remove_edges_to_node(node, keep_set=set(pre_nodes))
                            for pre in pre_nodes:
                                # 添加节点
                                dequantize_func = f.Dequantize(scale=scale, zero_point=zero_point)
                                tensor = dequantize_func(pre.true_obj)
                                dequantize_node = graph.add_node(dequantize_func)
                                tensor_node = graph.add_node(weakref.ref(tensor))
                                tensor_node.quantized = False
                                tensor_node.dequantized = True
                                # 添加边
                                graph.add_edge(pre, dequantize_node)
                                graph.add_edge(dequantize_node, tensor_node)
                                graph.add_edge(tensor_node, node)
                    
                        # 标记后继节点
                        for suc in suc_nodes:
                            suc.quantized = False
                            suc.dequantized = True

#"""