package orchestrator

import (
	"fmt"
	"sync"
	"time"
)

type JobPriority int

const (
	PriorityLow JobPriority = iota
	PriorityNormal
	PriorityHigh
	PriorityCritical
)

type JobState string

const (
	JobPending   JobState = "pending"
	JobScheduled JobState = "scheduled"
	JobRunning   JobState = "running"
	JobFailed    JobState = "failed"
	JobCompleted JobState = "completed"
)

type TrainingJob struct {
	ID           string
	Name         string
	Priority     JobPriority
	State        JobState
	GPUsRequired int
	Model        string
	Dataset      string
	Config       TrainingConfig
	CreatedAt    time.Time
	StartedAt    *time.Time
	CompletedAt  *time.Time
	NodeIDs      []string
	Error        string
}

type TrainingConfig struct {
	BatchSize             int
	GradientAccumulation  int
	LearningRate          float64
	MaxSteps              int
	CheckpointEverySteps  int
	EvalEverySteps        int
	FSDPShardingStrategy  string
	MixedPrecision        string
}

type Scheduler struct {
	mu       sync.Mutex
	jobs     map[string]*TrainingJob
	queue    []*TrainingJob
	gpuPool  *GPUPool
}

type GPUPool struct {
	mu    sync.Mutex
	nodes map[string]*GPUNode
}

type GPUNode struct {
	ID           string
	InstanceType string
	GPUs         []GPU
	Available    bool
	LastHealthAt time.Time
}

type GPU struct {
	Index     int
	Model     string
	MemoryMB  int
	PCIeGen   int
	NVLink    bool
	Healthy   bool
	InUse     bool
}

func NewScheduler() *Scheduler {
	return &Scheduler{
		jobs:    make(map[string]*TrainingJob),
		gpuPool: &GPUPool{nodes: make(map[string]*GPUNode)},
	}
}

func (s *Scheduler) Submit(job *TrainingJob) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	if job.GPUsRequired < 1 {
		return fmt.Errorf("job requires at least 1 GPU")
	}

	job.State = JobPending
	job.CreatedAt = time.Now()
	s.jobs[job.ID] = job
	s.queue = append(s.queue, job)
	s.sortQueue()

	return nil
}

func (s *Scheduler) Schedule() (*TrainingJob, error) {
	s.mu.Lock()
	defer s.mu.Unlock()

	for _, job := range s.queue {
		if job.State != JobPending {
			continue
		}

		nodes, err := s.gpuPool.Allocate(job.GPUsRequired)
		if err != nil {
			continue
		}

		job.State = JobScheduled
		job.NodeIDs = nodes
		return job, nil
	}

	return nil, fmt.Errorf("no pending jobs can be scheduled with available GPUs")
}

func (s *Scheduler) sortQueue() {
	// Priority-based sorting: higher priority first, then FIFO within same priority
	for i := 1; i < len(s.queue); i++ {
		for j := i; j > 0 && s.queue[j].Priority > s.queue[j-1].Priority; j-- {
			s.queue[j], s.queue[j-1] = s.queue[j-1], s.queue[j]
		}
	}
}

func (p *GPUPool) RegisterNode(node *GPUNode) {
	p.mu.Lock()
	defer p.mu.Unlock()
	p.nodes[node.ID] = node
}

func (p *GPUPool) Allocate(gpusNeeded int) ([]string, error) {
	p.mu.Lock()
	defer p.mu.Unlock()

	var allocated []string
	remaining := gpusNeeded

	for _, node := range p.nodes {
		if !node.Available {
			continue
		}
		available := 0
		for _, gpu := range node.GPUs {
			if gpu.Healthy && !gpu.InUse {
				available++
			}
		}
		if available > 0 {
			allocated = append(allocated, node.ID)
			remaining -= available
			if remaining <= 0 {
				return allocated, nil
			}
		}
	}

	return nil, fmt.Errorf("need %d GPUs, only %d available", gpusNeeded, gpusNeeded-remaining)
}
