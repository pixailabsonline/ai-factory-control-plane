package orchestrator

import (
	"fmt"
	"log"
	"time"
)

type RecoveryManager struct {
	scheduler        *Scheduler
	checkpointFinder CheckpointFinder
	maxRetries       int
	cooldownPeriod   time.Duration
}

type CheckpointFinder interface {
	LatestValid(jobID string) (string, int, error) // returns path, step, error
}

type RecoveryEvent struct {
	JobID           string
	FailedAt        time.Time
	RecoveredAt     *time.Time
	FailureReason   string
	CheckpointPath  string
	CheckpointStep  int
	StepsLost       int
	RecoveryTimeSec float64
	Attempt         int
	Success         bool
}

func NewRecoveryManager(scheduler *Scheduler, finder CheckpointFinder) *RecoveryManager {
	return &RecoveryManager{
		scheduler:        scheduler,
		checkpointFinder: finder,
		maxRetries:       3,
		cooldownPeriod:   30 * time.Second,
	}
}

func (rm *RecoveryManager) HandleFailure(job *TrainingJob, reason string) (*RecoveryEvent, error) {
	event := &RecoveryEvent{
		JobID:         job.ID,
		FailedAt:      time.Now(),
		FailureReason: reason,
	}

	log.Printf("[recovery] job %s failed: %s", job.ID, reason)

	rm.scheduler.mu.Lock()
	job.State = JobFailed
	job.Error = reason
	rm.scheduler.mu.Unlock()

	ckptPath, ckptStep, err := rm.checkpointFinder.LatestValid(job.ID)
	if err != nil {
		event.Success = false
		return event, fmt.Errorf("no valid checkpoint found for job %s: %w", job.ID, err)
	}

	event.CheckpointPath = ckptPath
	event.CheckpointStep = ckptStep

	if job.Config.MaxSteps > 0 {
		event.StepsLost = estimateStepsLost(job, ckptStep)
	}

	log.Printf("[recovery] job %s: found checkpoint at step %d (%s), lost ~%d steps",
		job.ID, ckptStep, ckptPath, event.StepsLost)

	time.Sleep(rm.cooldownPeriod)

	rm.scheduler.mu.Lock()
	job.State = JobPending
	job.Error = ""
	rm.scheduler.mu.Unlock()

	now := time.Now()
	event.RecoveredAt = &now
	event.RecoveryTimeSec = now.Sub(event.FailedAt).Seconds()
	event.Success = true

	log.Printf("[recovery] job %s re-queued, recovery took %.1fs", job.ID, event.RecoveryTimeSec)

	return event, nil
}

func estimateStepsLost(job *TrainingJob, lastCheckpointStep int) int {
	if job.Config.CheckpointEverySteps == 0 {
		return 0
	}
	return job.Config.CheckpointEverySteps
}
