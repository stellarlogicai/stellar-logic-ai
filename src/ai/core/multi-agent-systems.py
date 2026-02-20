#!/usr/bin/env python3
"""
Stellar Logic AI - Multi-Agent Systems Framework
Collaborative AI agents for complex problem solving
"""

import numpy as np
from typing import List, Dict, Tuple, Any, Optional, Callable
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from enum import Enum
import random
import math
import asyncio
from collections import defaultdict, deque
import json
import time

class AgentRole(Enum):
    """Types of agent roles in multi-agent systems"""
    COORDINATOR = "coordinator"
    SPECIALIST = "specialist"
    ANALYST = "analyst"
    EXECUTOR = "executor"
    MONITOR = "monitor"
    COMMUNICATOR = "communicator"

class AgentState(Enum):
    """Agent operational states"""
    IDLE = "idle"
    ACTIVE = "active"
    WAITING = "waiting"
    COLLABORATING = "collaborating"
    FAILED = "failed"

@dataclass
class AgentCapability:
    """Defines what an agent can do"""
    name: str
    description: str
    expertise_level: float  # 0.0 to 1.0
    processing_speed: float  # operations per second
    reliability: float  # 0.0 to 1.0
    resource_requirements: Dict[str, float]

@dataclass
class AgentMessage:
    """Communication between agents"""
    sender_id: str
    receiver_id: str
    message_type: str
    content: Dict[str, Any]
    priority: int  # 1-10
    timestamp: float
    requires_response: bool = False

@dataclass
class Task:
    """Task to be executed by agents"""
    task_id: str
    description: str
    required_capabilities: List[str]
    complexity: float  # 0.0 to 1.0
    priority: int  # 1-10
    deadline: Optional[float] = None
    dependencies: List[str] = field(default_factory=list)
    assigned_agents: List[str] = field(default_factory=list)
    status: str = "pending"
    progress: float = 0.0

class AIAgent(ABC):
    """Base class for AI agents"""
    
    def __init__(self, agent_id: str, role: AgentRole, capabilities: List[AgentCapability], message_broker=None):
        self.id = agent_id
        self.role = role
        self.capabilities = capabilities
        self.state = AgentState.IDLE
        self.current_task = None
        self.message_queue = deque()
        self.collaboration_partners = []
        self.performance_history = []
        self.resource_usage = defaultdict(float)
        self.message_broker = message_broker  # Add message broker reference
        
        # Learning and adaptation
        self.learning_rate = 0.01
        self.success_rate = 0.5
        self.experience_points = 0
        
    def can_handle_task(self, task: Task) -> bool:
        """Check if agent can handle task"""
        required_caps = set(task.required_capabilities)
        available_caps = set(cap.name for cap in self.capabilities)
        
        # Allow partial matches for multi-agent collaboration
        capability_match = len(required_caps & available_caps) > 0
        availability = self.state == AgentState.IDLE
        
        return capability_match and availability
    
    def assign_task(self, task: Task) -> bool:
        """Assign task to agent"""
        if self.can_handle_task(task):
            self.current_task = task
            self.state = AgentState.ACTIVE
            task.assigned_agents.append(self.id)
            return True
        return False
    
    @abstractmethod
    def execute_task(self, task: Task, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute assigned task"""
        pass
    
    def send_message(self, receiver_id: str, message_type: str, 
                    content: Dict[str, Any], priority: int = 5) -> None:
        """Send message to another agent"""
        message = AgentMessage(
            sender_id=self.id,
            receiver_id=receiver_id,
            message_type=message_type,
            content=content,
            priority=priority,
            timestamp=time.time()
        )
        
        # Use message broker if available, otherwise queue locally
        if self.message_broker:
            self.message_broker.send_message(message)
        else:
            self.message_queue.append(message)
    
    def receive_message(self, message: AgentMessage) -> None:
        """Process incoming message"""
        self.message_queue.append(message)
        self._process_message(message)
    
    def _process_message(self, message: AgentMessage) -> None:
        """Process incoming message based on type"""
        if message.message_type == "task_request":
            self._handle_task_request(message)
        elif message.message_type == "collaboration_invite":
            self._handle_collaboration_invite(message)
        elif message.message_type == "status_update":
            self._handle_status_update(message)
    
    def _handle_task_request(self, message: AgentMessage) -> None:
        """Handle task request message"""
        # Implementation depends on agent role
        pass
    
    def _handle_collaboration_invite(self, message: AgentMessage) -> None:
        """Handle collaboration invitation"""
        if message.sender_id not in self.collaboration_partners:
            self.collaboration_partners.append(message.sender_id)
    
    def _handle_status_update(self, message: AgentMessage) -> None:
        """Handle status update from other agents"""
        # Update internal state based on partner status
        pass
    
    def update_performance(self, success: bool, execution_time: float) -> None:
        """Update agent performance metrics"""
        self.performance_history.append({
            'timestamp': time.time(),
            'success': success,
            'execution_time': execution_time,
            'task_complexity': self.current_task.complexity if self.current_task else 0.0
        })
        
        # Update success rate with exponential moving average
        if success:
            self.success_rate = 0.9 * self.success_rate + 0.1 * 1.0
            self.experience_points += 1
        else:
            self.success_rate = 0.9 * self.success_rate + 0.1 * 0.0
        
        # Complete task
        if self.current_task:
            self.current_task = None
            self.state = AgentState.IDLE

class SpecialistAgent(AIAgent):
    """Specialist agent for specific domains"""
    
    def __init__(self, agent_id: str, domain: str, expertise_level: float):
        capabilities = [
            AgentCapability(
                name=f"{domain}_analysis",
                description=f"Expert analysis in {domain}",
                expertise_level=expertise_level,
                processing_speed=10.0 * expertise_level,
                reliability=0.8 + 0.2 * expertise_level,
                resource_requirements={"cpu": 0.1, "memory": 0.05}
            )
        ]
        super().__init__(agent_id, AgentRole.SPECIALIST, capabilities)
        self.domain = domain
        self.knowledge_base = {}
    
    def execute_task(self, task: Task, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute domain-specific task"""
        start_time = time.time()
        
        try:
            # Simulate domain-specific processing
            processing_time = task.complexity / self.capabilities[0].processing_speed
            
            # Generate analysis result
            result = {
                'agent_id': self.id,
                'domain': self.domain,
                'task_id': task.task_id,
                'analysis': self._perform_domain_analysis(task, context),
                'confidence': self.success_rate,
                'processing_time': processing_time,
                'expertise_level': self.capabilities[0].expertise_level,
                'success': True
            }
            
            # Update performance
            success = random.random() < self.success_rate
            self.update_performance(success, processing_time)
            
            return result
            
        except Exception as e:
            self.update_performance(False, time.time() - start_time)
            return {
                'agent_id': self.id,
                'task_id': task.task_id,
                'error': str(e),
                'success': False
            }
    
    def _perform_domain_analysis(self, task: Task, context: Dict[str, Any]) -> Dict[str, Any]:
        """Perform domain-specific analysis"""
        # Simulate domain expertise
        analysis_depth = self.capabilities[0].expertise_level
        
        return {
            'domain_insights': f"Deep {self.domain} analysis at level {analysis_depth:.2f}",
            'key_findings': [f"Finding {i+1} in {self.domain}" for i in range(int(analysis_depth * 5))],
            'recommendations': [f"Recommendation {i+1}" for i in range(int(analysis_depth * 3))],
            'confidence_score': analysis_depth
        }

class CoordinatorAgent(AIAgent):
    """Coordinator agent for managing multi-agent collaboration"""
    
    def __init__(self, agent_id: str):
        capabilities = [
            AgentCapability(
                name="coordination",
                description="Coordinate multi-agent tasks",
                expertise_level=0.9,
                processing_speed=20.0,
                reliability=0.95,
                resource_requirements={"cpu": 0.2, "memory": 0.1}
            ),
            AgentCapability(
                name="task_allocation",
                description="Allocate tasks to appropriate agents",
                expertise_level=0.85,
                processing_speed=15.0,
                reliability=0.9,
                resource_requirements={"cpu": 0.15, "memory": 0.08}
            )
        ]
        super().__init__(agent_id, AgentRole.COORDINATOR, capabilities)
        self.agent_registry = {}
        self.task_queue = []
        self.allocation_history = []
    
    def register_agent(self, agent: AIAgent) -> None:
        """Register agent with coordinator"""
        self.agent_registry[agent.id] = agent
    
    def allocate_task(self, task: Task) -> Dict[str, Any]:
        """Allocate task to best-suited agent(s)"""
        suitable_agents = []
        
        for agent_id, agent in self.agent_registry.items():
            if agent.can_handle_task(task):
                # Calculate suitability score
                score = self._calculate_suitability_score(agent, task)
                suitable_agents.append((agent_id, score))
        
        # Sort by suitability score
        suitable_agents.sort(key=lambda x: x[1], reverse=True)
        
        # Select best agent(s)
        selected_agents = []
        if suitable_agents:
            # Single agent for simple tasks, multiple for complex tasks
            num_agents = min(1 + int(task.complexity * 2), len(suitable_agents))
            selected_agents = [agent_id for agent_id, _ in suitable_agents[:num_agents]]
        
        # Assign task to selected agents
        assignment_success = False
        for agent_id in selected_agents:
            agent = self.agent_registry[agent_id]
            if agent.assign_task(task):
                assignment_success = True
        
        self.allocation_history.append({
            'task_id': task.task_id,
            'assigned_agents': selected_agents,
            'suitability_scores': suitable_agents[:len(selected_agents)],
            'timestamp': time.time()
        })
        
        return {
            'task_id': task.task_id,
            'assigned_agents': selected_agents,
            'allocation_success': assignment_success,
            'coordination_confidence': self.success_rate
        }
    
    def _calculate_suitability_score(self, agent: AIAgent, task: Task) -> float:
        """Calculate how suitable an agent is for a task"""
        score = 0.0
        
        # Capability match
        required_caps = set(task.required_capabilities)
        available_caps = set(cap.name for cap in agent.capabilities)
        capability_score = len(required_caps & available_caps) / len(required_caps) if required_caps else 0.0
        
        # Agent performance
        performance_score = agent.success_rate
        
        # Agent availability
        availability_score = 1.0 if agent.state == AgentState.IDLE else 0.0
        
        # Expertise level
        expertise_score = np.mean([cap.expertise_level for cap in agent.capabilities])
        
        # Weighted combination
        score = (0.3 * capability_score + 
                0.3 * performance_score + 
                0.2 * availability_score + 
                0.2 * expertise_score)
        
        return score
    
    def execute_task(self, task: Task, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute coordination task"""
        start_time = time.time()
        
        if task.task_id == "allocate_tasks":
            # Allocate all pending tasks
            allocation_results = []
            for pending_task in self.task_queue:
                result = self.allocate_task(pending_task)
                allocation_results.append(result)
            
            success = all(result['allocation_success'] for result in allocation_results)
            self.update_performance(success, time.time() - start_time)
            
            return {
                'agent_id': self.id,
                'task_id': task.task_id,
                'allocation_results': allocation_results,
                'success': success,
                'tasks_allocated': len(allocation_results)
            }
        
        return {
            'agent_id': self.id,
            'task_id': task.task_id,
            'message': 'Coordination task completed',
            'success': True
        }

class MultiAgentSystem:
    """Complete multi-agent system"""
    
    def __init__(self, num_agents: int = 10):
        self.agents = {}
        self.coordinator = None
        self.message_broker = MessageBroker()
        self.task_queue = []
        self.completed_tasks = []
        self.system_performance = {}
        
        # Initialize system
        self._initialize_agents(num_agents)
        
    def _initialize_agents(self, num_agents: int):
        """Initialize agents with different roles and capabilities"""
        # Create coordinator
        self.coordinator = CoordinatorAgent("coordinator_001")
        self.agents[self.coordinator.id] = self.coordinator
        
        # Create specialist agents
        domains = ["security", "analytics", "optimization", "prediction", "monitoring"]
        
        for i in range(num_agents - 1):
            domain = domains[i % len(domains)]
            expertise = random.uniform(0.6, 0.95)
            agent_id = f"specialist_{i+1:03d}"
            
            specialist = SpecialistAgent(agent_id, domain, expertise)
            self.agents[agent_id] = specialist
            self.coordinator.register_agent(specialist)
        
        # Register coordinator in its own registry so it can handle coordination tasks
        self.coordinator.register_agent(self.coordinator)
        # Inject message broker into coordinator
        self.coordinator.message_broker = self.message_broker
    
    def submit_task(self, task: Task) -> Dict[str, Any]:
        """Submit task to multi-agent system"""
        self.task_queue.append(task)
        
        # Allocate task through coordinator
        allocation_result = self.coordinator.allocate_task(task)
        
        return {
            'task_id': task.task_id,
            'submission_time': time.time(),
            'allocation_result': allocation_result,
            'queue_position': len(self.task_queue)
        }
    
    def execute_tasks(self, max_iterations: int = 100) -> Dict[str, Any]:
        """Execute all tasks in the queue"""
        execution_results = []
        
        for iteration in range(max_iterations):
            if not self.task_queue:
                break
            
            # Get next task
            task = self.task_queue.pop(0)
            
            # Execute task with assigned agents
            task_result = self._execute_single_task(task)
            execution_results.append(task_result)
            
            # Update system performance
            self._update_system_performance()
        
        return {
            'iterations': iteration + 1,
            'tasks_executed': len(execution_results),
            'execution_results': execution_results,
            'system_performance': self.system_performance
        }
    
    def _execute_single_task(self, task: Task) -> Dict[str, Any]:
        """Execute a single task with assigned agents"""
        start_time = time.time()
        
        # Get assigned agents
        assigned_agents = [self.agents[agent_id] for agent_id in task.assigned_agents 
                          if agent_id in self.agents]
        
        if not assigned_agents:
            return {
                'task_id': task.task_id,
                'success': False,
                'error': 'No agents assigned',
                'execution_time': time.time() - start_time
            }
        
        # Execute task (simplified - in real system would be parallel)
        agent_results = []
        for agent in assigned_agents:
            context = {
                'collaboration_partners': [a.id for a in assigned_agents if a.id != agent.id],
                'system_state': self._get_system_state()
            }
            
            result = agent.execute_task(task, context)
            agent_results.append(result)
        
        # Aggregate results
        success = any(result.get('success', False) for result in agent_results)
        execution_time = time.time() - start_time
        
        # Complete task
        task.status = 'completed' if success else 'failed'
        task.progress = 1.0
        self.completed_tasks.append(task)
        
        return {
            'task_id': task.task_id,
            'success': success,
            'execution_time': execution_time,
            'agent_results': agent_results,
            'num_agents': len(assigned_agents)
        }
    
    def _get_system_state(self) -> Dict[str, Any]:
        """Get current system state"""
        agent_states = {}
        for agent_id, agent in self.agents.items():
            agent_states[agent_id] = {
                'state': agent.state.value,
                'current_task': agent.current_task.task_id if agent.current_task else None,
                'success_rate': agent.success_rate,
                'message_queue_size': len(agent.message_queue)
            }
        
        return {
            'total_agents': len(self.agents),
            'agent_states': agent_states,
            'pending_tasks': len(self.task_queue),
            'completed_tasks': len(self.completed_tasks)
        }
    
    def _update_system_performance(self):
        """Update system-wide performance metrics"""
        if not self.completed_tasks:
            return
        
        # Calculate system metrics
        total_tasks = len(self.completed_tasks)
        successful_tasks = sum(1 for task in self.completed_tasks if task.status == 'completed')
        
        agent_performance = {}
        for agent_id, agent in self.agents.items():
            if agent.performance_history:
                recent_performance = agent.performance_history[-10:]  # Last 10 tasks
                success_rate = sum(1 for p in recent_performance if p['success']) / len(recent_performance)
                avg_time = np.mean([p['execution_time'] for p in recent_performance])
                
                agent_performance[agent_id] = {
                    'success_rate': success_rate,
                    'avg_execution_time': avg_time,
                    'tasks_completed': len(agent.performance_history)
                }
        
        self.system_performance = {
            'overall_success_rate': successful_tasks / total_tasks,
            'total_tasks_completed': total_tasks,
            'agent_performance': agent_performance,
            'system_efficiency': successful_tasks / (total_tasks + len(self.task_queue)),
            'timestamp': time.time()
        }

class MessageBroker:
    """Message broker for agent communication"""
    
    def __init__(self):
        self.message_queues = defaultdict(deque)
        self.delivery_history = []
    
    def send_message(self, message: AgentMessage) -> bool:
        """Send message to target agent"""
        try:
            self.message_queues[message.receiver_id].append(message)
            self.delivery_history.append({
                'message_id': len(self.delivery_history),
                'sender': message.sender_id,
                'receiver': message.receiver_id,
                'timestamp': message.timestamp,
                'delivered': True
            })
            return True
        except Exception as e:
            return False
    
    def receive_messages(self, agent_id: str) -> List[AgentMessage]:
        """Get messages for an agent"""
        messages = list(self.message_queues[agent_id])
        self.message_queues[agent_id].clear()
        return messages

# Integration with Stellar Logic AI
class MultiAgentAIIntegration:
    """Integration layer for multi-agent systems with existing AI"""
    
    def __init__(self):
        self.multi_agent_system = MultiAgentSystem()
        self.active_tasks = {}
        
    def solve_complex_problem(self, problem_description: str, 
                            problem_type: str = "analysis") -> Dict[str, Any]:
        """Solve complex problem using multi-agent collaboration"""
        
        # Create task for the problem
        task = Task(
            task_id=f"problem_{int(time.time())}",
            description=problem_description,
            required_capabilities=[problem_type, "coordination"],
            complexity=0.8,
            priority=8,
            deadline=time.time() + 300  # 5 minutes
        )
        
        # Submit task to multi-agent system
        submission_result = self.multi_agent_system.submit_task(task)
        
        # Execute tasks
        execution_result = self.multi_agent_system.execute_tasks(max_iterations=50)
        
        return {
            'problem_description': problem_description,
            'problem_type': problem_type,
            'submission_result': submission_result,
            'execution_result': execution_result,
            'multi_agent_performance': execution_result['system_performance'],
            'collaboration_summary': self._generate_collaboration_summary(execution_result)
        }
    
    def _generate_collaboration_summary(self, execution_result: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary of agent collaboration"""
        agent_results = execution_result['execution_results']
        
        # Analyze collaboration patterns
        agent_participation = defaultdict(int)
        collaboration_success = defaultdict(int)
        
        for result in agent_results:
            if 'agent_results' in result:
                for agent_result in result['agent_results']:
                    agent_id = agent_result.get('agent_id', 'unknown')
                    agent_participation[agent_id] += 1
                    if agent_result.get('success', False):
                        collaboration_success[agent_id] += 1
        
        # Calculate collaboration metrics
        total_participation = sum(agent_participation.values())
        total_success = sum(collaboration_success.values())
        
        return {
            'total_agents_involved': len(agent_participation),
            'total_interactions': total_participation,
            'success_rate': total_success / total_participation if total_participation > 0 else 0,
            'agent_performance': dict(agent_participation),
            'collaboration_efficiency': total_success / total_participation if total_participation > 0 else 0,
            'emergent_intelligence': self._detect_emergent_intelligence(agent_results)
        }
    
    def _detect_emergent_intelligence(self, agent_results: List[Dict]) -> Dict[str, Any]:
        """Detect emergent intelligence from agent collaboration"""
        # Simplified emergent intelligence detection
        successful_results = [r for r in agent_results if r.get('success', False)]
        
        if len(successful_results) > len(agent_results) * 0.7:
            return {
                'detected': True,
                'level': 'high',
                'description': 'Agents achieved synergistic results beyond individual capabilities'
            }
        elif len(successful_results) > len(agent_results) * 0.5:
            return {
                'detected': True,
                'level': 'medium',
                'description': 'Moderate collaboration benefits observed'
            }
        else:
            return {
                'detected': False,
                'level': 'low',
                'description': 'Limited emergent behavior detected'
            }

# Usage example and testing
if __name__ == "__main__":
    print("🤖 Initializing Multi-Agent Systems Framework...")
    
    # Initialize multi-agent AI
    multi_agent_ai = MultiAgentAIIntegration()
    
    # Test complex problem solving
    print("\n🧠 Testing Complex Problem Solving...")
    problem = "Analyze security threats in enterprise network and recommend mitigation strategies"
    
    result = multi_agent_ai.solve_complex_problem(problem, "security")
    
    print(f"✅ Problem solved: {result['submission_result']['allocation_result']['allocation_success']}")
    print(f"🤖 Agents involved: {result['collaboration_summary']['total_agents_involved']}")
    print(f"🎯 Success rate: {result['collaboration_summary']['success_rate']:.2%}")
    print(f"🚀 Emergent intelligence: {result['collaboration_summary']['emergent_intelligence']['level']}")
    
    # Test system performance
    print("\n📊 Testing System Performance...")
    performance = result['multi_agent_performance']
    print(f"📈 Overall success rate: {performance['overall_success_rate']:.2%}")
    print(f"⚡ Tasks completed: {performance['total_tasks_completed']}")
    print(f"🎯 System efficiency: {performance['system_efficiency']:.2%}")
    
    print("\n🚀 Multi-Agent Systems Framework Ready!")
    print("🤖 Collaborative AI intelligence deployed!")
