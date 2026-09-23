/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */
import React, { useCallback, useMemo } from 'react';
import {
  ReactFlow,
  Background,
  Controls,
  MiniMap,
  Panel,
  addEdge,
  applyNodeChanges,
  applyEdgeChanges,
  type Node as FlowNode,
  type Edge,
  type NodeChange,
  type EdgeChange,
  type Connection,
} from '@xyflow/react';
import { motion, AnimatePresence } from 'motion/react';
import {
  Activity, BarChart3, BrainCircuit, Clock, Cloud, Cpu, Database, Globe,
  Layers, Pause, Play, Plus, Save, Settings, Shield, Sparkles, Trash2,
  TrendingUp, Zap,
} from 'lucide-react';

import { cn } from '../../lib/utils';
import { FilterNode, PositionSizerNode, RiskManagementNode, KillswitchNode } from '../WorkflowNodes';
import { CustomNode } from './CustomNode';
import { NodeProperties } from './NodeProperties';
import { DraggableComponent } from './DraggableComponent';

const nodeTypes = {
  workflow: CustomNode,
  filter: FilterNode,
  positionSizer: PositionSizerNode,
  riskManagement: RiskManagementNode,
  killswitch: KillswitchNode,
};

export interface Workflow {
  id: string;
  name: string;
  nodes: FlowNode[];
  edges: Edge[];
  isRunning: boolean;
}

export interface WorkflowBuilderProps {
  nodes: FlowNode[];
  edges: Edge[];
  setNodes: React.Dispatch<React.SetStateAction<FlowNode[]>>;
  setEdges: React.Dispatch<React.SetStateAction<Edge[]>>;
  workflows: Workflow[];
  activeWorkflowId: string | null;
  isZenMode: boolean;
  isOptimizing: boolean;
  createNewWorkflow: () => void;
  loadWorkflow: (id: string) => void;
  toggleWorkflowRun: (id: string) => void;
  saveWorkflow: () => void;
  addNewNode: () => void;
  handleWorkflowOptimization: () => void;
  executeNodeTrade: (node: FlowNode) => void;
}

/**
 * The @xyflow/react canvas and its side panels.
 *
 * This component exists so the ~173 kB flow library stays out of the entry
 * chunk: it is only fetched when the user opens Agent Builder. Graph state
 * still lives in App (the Backtesting view reads it), so it is passed in --
 * but every xyflow *runtime* import is confined to this module.
 */
export default function WorkflowBuilder({
  nodes,
  edges,
  setNodes,
  setEdges,
  workflows,
  activeWorkflowId,
  isZenMode,
  isOptimizing,
  createNewWorkflow,
  loadWorkflow,
  toggleWorkflowRun,
  saveWorkflow,
  addNewNode,
  handleWorkflowOptimization,
  executeNodeTrade,
}: WorkflowBuilderProps) {
  // Equivalent to useNodesState/useEdgesState, but driven by the state App owns.
  const onNodesChange = useCallback(
    (changes: NodeChange<FlowNode>[]) => setNodes((nds) => applyNodeChanges(changes, nds)),
    [setNodes]
  );
  const onEdgesChange = useCallback(
    (changes: EdgeChange<Edge>[]) => setEdges((eds) => applyEdgeChanges(changes, eds)),
    [setEdges]
  );
  const onConnect = useCallback(
    (params: Connection) =>
      setEdges((eds) => addEdge({ ...params, animated: true, style: { stroke: '#10b981' } } as any, eds)),
    [setEdges]
  );

  const selectedNodes = useMemo(() => nodes.filter((n) => n.selected), [nodes]);

  const updateNodeConfig = useCallback((nodeId: string, config: any) => {
    setNodes((nds) => nds.map((node) =>
      node.id === nodeId ? { ...node, data: { ...node.data, config } } : node
    ));
  }, [setNodes]);

  const updateNodeData = useCallback((nodeId: string, newData: any) => {
    setNodes((nds) => nds.map((node) =>
      node.id === nodeId ? { ...node, data: { ...node.data, ...newData } } : node
    ));
  }, [setNodes]);

  const handleBatchDelete = useCallback(() => {
    const selectedIds = new Set(selectedNodes.map((n) => n.id));
    setNodes((nds) => nds.filter((n) => !selectedIds.has(n.id)));
    setEdges((eds) => eds.filter((e) => !selectedIds.has(e.source) && !selectedIds.has(e.target)));
  }, [selectedNodes, setNodes, setEdges]);

  const handleBatchColorChange = useCallback((color: string) => {
    setNodes((nds) => nds.map((node) =>
      selectedNodes.find((sn) => sn.id === node.id)
        ? { ...node, data: { ...node.data, color } }
        : node
    ));
  }, [selectedNodes, setNodes]);

  const handleGroupNodes = useCallback(() => {
    if (selectedNodes.length < 2) return;

    const minX = Math.min(...selectedNodes.map((n) => n.position.x));
    const minY = Math.min(...selectedNodes.map((n) => n.position.y));
    const maxX = Math.max(...selectedNodes.map((n) => n.position.x + 180)); // Approx width
    const maxY = Math.max(...selectedNodes.map((n) => n.position.y + 60));  // Approx height

    const groupId = `group-${Date.now()}`;
    const groupNode: FlowNode = {
      id: groupId,
      type: 'group',
      data: { label: 'New Group' },
      position: { x: minX - 20, y: minY - 40 },
      style: {
        width: maxX - minX + 40,
        height: maxY - minY + 60,
        backgroundColor: 'rgba(16, 185, 129, 0.05)',
        border: '1px dashed #10b981',
        borderRadius: '12px',
      },
    };

    setNodes((nds) => [
      ...nds.map((node) => {
        if (selectedNodes.find((sn) => sn.id === node.id)) {
          return {
            ...node,
            parentId: groupId,
            extent: 'parent' as const,
            position: {
              x: node.position.x - (minX - 20),
              y: node.position.y - (minY - 40),
            },
          };
        }
        return node;
      }),
      groupNode,
    ]);
  }, [selectedNodes, setNodes]);

  return (
      <ReactFlow
        nodes={nodes}
        edges={edges}
        onNodesChange={onNodesChange}
        onEdgesChange={onEdgesChange}
        onConnect={onConnect}
        nodeTypes={nodeTypes}
        fitView
        className="bg-[#0A0A0B]"
      >
        <Background color="#27272a" gap={20} />
        <Controls className="bg-zinc-900 border-zinc-800 fill-white" />
        <MiniMap
          nodeColor="#10b981"
          maskColor="rgba(0,0,0,0.5)"
          className="bg-zinc-900 border border-zinc-800 rounded-xl"
        />

        <Panel position="top-left" className={cn("flex flex-col gap-2 transition-all duration-300 max-h-[80vh] w-[260px]", isZenMode && "opacity-0 pointer-events-none -translate-x-10")}>
          <div className="bg-[#141416] border border-zinc-800 rounded-xl p-4 shadow-2xl flex-shrink-0">
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-sm font-bold flex items-center gap-2">
                <Layers size={16} className="text-emerald-400" /> Workflows
              </h3>
              <button
                onClick={createNewWorkflow}
                className="p-1 hover:bg-white/5 rounded-md text-zinc-500 hover:text-emerald-400 transition-colors"
              >
                <Plus size={16} />
              </button>
            </div>
            <div className="space-y-2 max-h-[150px] overflow-y-auto pr-2 scrollbar-default">
              {workflows.length === 0 && (
                <p className="text-[10px] text-zinc-500 italic text-center py-4">No saved workflows</p>
              )}
              {workflows.map(wf => (
                <div
                  key={wf.id}
                  className={cn(
                    "group flex items-center justify-between p-2 rounded-lg border transition-all cursor-pointer",
                    activeWorkflowId === wf.id ? "bg-emerald-500/10 border-emerald-500/30" : "bg-zinc-900/50 border-zinc-800 hover:border-zinc-700"
                  )}
                  onClick={() => loadWorkflow(wf.id)}
                >
                  <div className="flex items-center gap-2 overflow-hidden">
                    <div className={cn(
                      "w-1.5 h-1.5 rounded-full flex-shrink-0",
                      wf.isRunning ? "bg-emerald-500 animate-pulse" : "bg-zinc-700"
                    )} />
                    <span className="text-xs font-medium truncate">{wf.name}</span>
                  </div>
                  <button
                    onClick={(e) => {
                      e.stopPropagation();
                      toggleWorkflowRun(wf.id);
                    }}
                    className={cn(
                      "p-1 rounded transition-colors",
                      wf.isRunning ? "text-rose-400 hover:bg-rose-500/10" : "text-emerald-400 hover:bg-emerald-500/10"
                    )}
                  >
                    {wf.isRunning ? <Pause size={12} /> : <Play size={12} />}
                  </button>
                </div>
              ))}
            </div>
          </div>

          <div className="bg-[#141416] border border-zinc-800 rounded-xl p-4 shadow-2xl flex-1 flex flex-col min-h-0 overflow-hidden">
            <h3 className="text-sm font-bold mb-4 flex-shrink-0 flex items-center gap-2">
              <Layers size={16} className="text-emerald-400" /> Components
            </h3>
            <div className="space-y-2 flex-1 overflow-y-auto min-h-0 pr-2 scrollbar-default">
              <DraggableComponent icon={<Activity size={14} />} label="RSI Indicator" type="Trigger" />
              <DraggableComponent icon={<Clock size={14} />} label="Time Delay" type="Condition" />
              <DraggableComponent icon={<Zap size={14} />} label="Market Buy" type="Action" />
              <DraggableComponent icon={<TrendingUp size={14} />} label="Take Profit" type="Condition" />
              <DraggableComponent icon={<Shield size={14} />} label="Stop Loss" type="Action" />
              <DraggableComponent icon={<Globe size={14} />} label="Webhook" type="Trigger" />
              <DraggableComponent icon={<Globe size={14} />} label="cTrader API" type="Integration" />
              <DraggableComponent icon={<Zap size={14} />} label="Binance API" type="Integration" />
              <DraggableComponent icon={<Database size={14} />} label="PostgreSQL" type="Integration" />
              <DraggableComponent icon={<Cloud size={14} />} label="Supabase" type="Integration" />
              <DraggableComponent icon={<BarChart3 size={14} />} label="Grafana" type="Integration" />
              <DraggableComponent icon={<Sparkles size={14} />} label="AI Agent" type="Action" />
            </div>
            <div className="pt-4 mt-2 border-t border-zinc-800 flex-shrink-0">
              <button
                onClick={addNewNode}
                className="w-full py-2.5 bg-emerald-500 hover:bg-emerald-400 text-black rounded-xl text-xs font-bold flex items-center justify-center gap-2 transition-all shadow-lg shadow-emerald-500/20"
              >
                <Plus size={16} /> Create New Node
              </button>
            </div>
          </div>
        </Panel>

        <Panel position="top-right" className={cn("flex flex-col gap-2 transition-all duration-300", isZenMode && "opacity-0 pointer-events-none translate-x-10")}>
          <div className="bg-[#141416] border border-zinc-800 rounded-xl p-4 shadow-2xl min-w-[200px]">
            <h3 className="text-sm font-bold mb-4 flex items-center gap-2">
              <Cpu size={16} className="text-emerald-400" /> Agent Status
            </h3>
            <div className="space-y-4">
              <div className="flex items-center justify-between">
                <span className="text-xs text-zinc-500">Status</span>
                <span className="text-xs font-bold text-emerald-400 flex items-center gap-1">
                  <div className="w-1.5 h-1.5 bg-emerald-500 rounded-full animate-pulse" /> Running
                </span>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-xs text-zinc-500">Total Trades</span>
                <span className="text-xs font-bold">142</span>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-xs text-zinc-500">Win Rate</span>
                <span className="text-xs font-bold text-emerald-400">68.4%</span>
              </div>
              <div className="pt-2 border-t border-zinc-800 flex flex-col gap-2">
                <button
                  onClick={handleWorkflowOptimization}
                  disabled={isOptimizing}
                  className="w-full bg-indigo-500/10 hover:bg-indigo-500/20 text-indigo-400 py-2 rounded-lg text-[10px] font-bold flex items-center justify-center gap-1 border border-indigo-500/20 transition-all"
                >
                  <BrainCircuit size={12} /> {isOptimizing ? 'Optimizing...' : 'Optimize with AI'}
                </button>
                <div className="flex gap-2">
                  <button className="flex-1 bg-zinc-800 hover:bg-zinc-700 py-2 rounded-lg text-[10px] font-bold flex items-center justify-center gap-1">
                    <Pause size={12} /> Pause
                  </button>
                  <button
                    onClick={saveWorkflow}
                    className="flex-1 bg-emerald-500 text-black py-2 rounded-lg text-[10px] font-bold flex items-center justify-center gap-1"
                  >
                    <Save size={12} /> Save
                  </button>
                </div>
              </div>
            </div>
          </div>

          {/* Node Properties Panel */}
          <AnimatePresence>
            {selectedNodes.length === 1 && (
              <motion.div
                initial={{ x: 50, opacity: 0 }}
                animate={{ x: 0, opacity: 1 }}
                exit={{ x: 50, opacity: 0 }}
                className="bg-[#141416] border border-zinc-800 rounded-xl p-4 shadow-2xl min-w-[260px] mt-2"
              >
                <h3 className="text-sm font-bold mb-4 flex items-center gap-2">
                  <Settings size={16} className="text-emerald-400" /> Node Properties
                </h3>
                <div className="space-y-4">
                  <div className="space-y-1.5">
                    <label className="text-[10px] uppercase text-zinc-500 font-bold">Label</label>
                    <input
                      type="text"
                      value={selectedNodes[0].data.label as string}
                      onChange={(e) => updateNodeData(selectedNodes[0].id, { label: e.target.value })}
                      aria-label="Node Label"
                      className="w-full bg-zinc-900 border border-zinc-800 rounded-lg py-2 px-3 text-xs focus:outline-none focus:border-emerald-500/50"
                    />
                  </div>
                  <div className="space-y-1.5">
                    <label className="text-[10px] uppercase text-zinc-500 font-bold">Type</label>
                    <select
                      value={selectedNodes[0].data.type as string}
                      onChange={(e) => updateNodeData(selectedNodes[0].id, { type: e.target.value })}
                      aria-label="Node Type"
                      className="w-full bg-zinc-900 border border-zinc-800 rounded-lg py-2 px-3 text-xs focus:outline-none focus:border-emerald-500/50 appearance-none"
                    >
                      <option value="Trigger">Trigger</option>
                      <option value="Condition">Condition</option>
                      <option value="Action">Action</option>
                      <option value="Integration">Integration</option>
                    </select>
                  </div>
                  <div className="space-y-1.5">
                    <label className="text-[10px] uppercase text-zinc-500 font-bold">Custom Logic</label>
                    <textarea
                      placeholder="Enter expression..."
                      aria-label="Custom Logic Expression"
                      className="w-full bg-zinc-900 border border-zinc-800 rounded-lg py-2 px-3 text-xs focus:outline-none focus:border-emerald-500/50 min-h-[80px] resize-none"
                    />
                  </div>
                </div>
              </motion.div>
            )}
          </AnimatePresence>
        </Panel>

        {/* Batch Actions Panel */}
        <AnimatePresence>
          {selectedNodes.length > 1 && (
            <Panel position="bottom-center">
              <motion.div
                initial={{ y: 50, opacity: 0 }}
                animate={{ y: 0, opacity: 1 }}
                exit={{ y: 50, opacity: 0 }}
                className="bg-[#141416] border border-emerald-500/50 rounded-2xl p-4 shadow-2xl flex items-center gap-6 mb-8"
              >
                <div className="flex items-center gap-3 pr-6 border-r border-zinc-800">
                  <div className="w-8 h-8 bg-emerald-500/20 rounded-lg flex items-center justify-center text-emerald-400 font-bold text-xs">
                    {selectedNodes.length}
                  </div>
                  <span className="text-sm font-semibold">Nodes Selected</span>
                </div>

                <div className="flex items-center gap-2">
                  <button
                    onClick={handleGroupNodes}
                    className="flex items-center gap-2 px-4 py-2 bg-zinc-800 hover:bg-zinc-700 rounded-xl text-xs font-bold transition-colors"
                  >
                    <Layers size={14} className="text-emerald-400" /> Group Selected
                  </button>

                  <div className="h-8 w-px bg-zinc-800 mx-2" />

                  <div className="flex items-center gap-1">
                    {['bg-blue-500/20', 'bg-amber-500/20', 'bg-emerald-500/20', 'bg-rose-500/20'].map(color => (
                      <button
                        key={color}
                        onClick={() => handleBatchColorChange(color)}
                        className={cn("w-6 h-6 rounded-full border border-zinc-800 hover:scale-110 transition-transform", color.replace('/20', ''))}
                        style={{ backgroundColor: color.includes('emerald') ? '#10b981' : color.includes('blue') ? '#3b82f6' : color.includes('amber') ? '#f59e0b' : '#f43f5e' }}
                      />
                    ))}
                  </div>

                  <div className="h-8 w-px bg-zinc-800 mx-2" />

                  <button
                    onClick={handleBatchDelete}
                    className="flex items-center gap-2 px-4 py-2 bg-rose-500/10 hover:bg-rose-500/20 text-rose-400 rounded-xl text-xs font-bold transition-colors"
                  >
                    <Trash2 size={14} /> Delete All
                  </button>
                </div>
              </motion.div>
            </Panel>
          )}
        </AnimatePresence>
        <Panel position="top-right" className={cn("flex flex-col gap-4 transition-all duration-300", isZenMode && "opacity-0 pointer-events-none translate-x-10")}>
          {selectedNodes.length === 1 && (selectedNodes[0].data.type === 'Integration' || selectedNodes[0].data.type === 'Action') && (
            <div className="space-y-4">
              <NodeProperties
                node={selectedNodes[0]}
                onUpdate={updateNodeConfig}
              />
              <button
                onClick={() => executeNodeTrade(selectedNodes[0])}
                className="w-full bg-emerald-500 hover:bg-emerald-400 text-black font-bold py-3 rounded-xl shadow-lg shadow-emerald-500/20 flex items-center justify-center gap-2 transition-all active:scale-95"
              >
                <Zap size={16} fill="currentColor" /> Execute Trade
              </button>
            </div>
          )}
        </Panel>
      </ReactFlow>
  );
}
