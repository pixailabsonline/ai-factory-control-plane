package cost

import (
	"fmt"
	"math"
	"time"
)

type InstancePricing struct {
	InstanceType string
	GPUCount     int
	GPUModel     string
	OnDemandPerHr float64
	SpotPerHr     float64
	FP16TFLOPS    float64
}

var KnownPricing = map[string]InstancePricing{
	"p3.2xlarge": {
		InstanceType:  "p3.2xlarge",
		GPUCount:      1,
		GPUModel:      "V100",
		OnDemandPerHr: 3.06,
		SpotPerHr:     0.92,
		FP16TFLOPS:    125,
	},
	"p3.8xlarge": {
		InstanceType:  "p3.8xlarge",
		GPUCount:      4,
		GPUModel:      "V100",
		OnDemandPerHr: 12.24,
		SpotPerHr:     3.67,
		FP16TFLOPS:    500,
	},
	"p3.16xlarge": {
		InstanceType:  "p3.16xlarge",
		GPUCount:      8,
		GPUModel:      "V100",
		OnDemandPerHr: 24.48,
		SpotPerHr:     7.34,
		FP16TFLOPS:    1000,
	},
	"p4d.24xlarge": {
		InstanceType:  "p4d.24xlarge",
		GPUCount:      8,
		GPUModel:      "A100",
		OnDemandPerHr: 32.77,
		SpotPerHr:     9.83,
		FP16TFLOPS:    2496,
	},
	"p5.48xlarge": {
		InstanceType:  "p5.48xlarge",
		GPUCount:      8,
		GPUModel:      "H100",
		OnDemandPerHr: 98.32,
		SpotPerHr:     29.50,
		FP16TFLOPS:    7916,
	},
}

type RunCost struct {
	JobID            string
	InstanceType     string
	StartTime        time.Time
	EndTime          *time.Time
	GPUHours         float64
	TotalCost        float64
	TokensTrained    int64
	CostPerMillionTokens float64
	WastedCost       float64
	WastedGPUHours   float64
}

type CostTracker struct {
	runs map[string]*RunCost
}

func NewCostTracker() *CostTracker {
	return &CostTracker{
		runs: make(map[string]*RunCost),
	}
}

func (ct *CostTracker) StartRun(jobID, instanceType string) error {
	pricing, ok := KnownPricing[instanceType]
	if !ok {
		return fmt.Errorf("unknown instance type: %s", instanceType)
	}

	_ = pricing
	ct.runs[jobID] = &RunCost{
		JobID:        jobID,
		InstanceType: instanceType,
		StartTime:    time.Now(),
	}

	return nil
}

func (ct *CostTracker) RecordTokens(jobID string, tokens int64) {
	run, ok := ct.runs[jobID]
	if !ok {
		return
	}
	run.TokensTrained += tokens
}

func (ct *CostTracker) RecordWaste(jobID string, wastedHours float64) {
	run, ok := ct.runs[jobID]
	if !ok {
		return
	}
	pricing := KnownPricing[run.InstanceType]
	run.WastedGPUHours += wastedHours * float64(pricing.GPUCount)
	run.WastedCost += wastedHours * pricing.OnDemandPerHr
}

func (ct *CostTracker) Finalize(jobID string) (*RunCost, error) {
	run, ok := ct.runs[jobID]
	if !ok {
		return nil, fmt.Errorf("unknown job: %s", jobID)
	}

	now := time.Now()
	run.EndTime = &now

	pricing := KnownPricing[run.InstanceType]
	wallHours := now.Sub(run.StartTime).Hours()
	run.GPUHours = wallHours * float64(pricing.GPUCount)
	run.TotalCost = wallHours * pricing.OnDemandPerHr

	if run.TokensTrained > 0 {
		run.CostPerMillionTokens = (run.TotalCost / float64(run.TokensTrained)) * 1_000_000
	}

	return run, nil
}

func ProjectCost(measuredTokensPerSec float64, measuredInstanceType string, targetParams int64, targetTokens int64, targetInstanceType string) string {
	measured := KnownPricing[measuredInstanceType]
	target := KnownPricing[targetInstanceType]

	scaleFactor := target.FP16TFLOPS / measured.FP16TFLOPS
	projectedTokensPerSec := measuredTokensPerSec * scaleFactor * 0.7 // 70% scaling efficiency

	totalSeconds := float64(targetTokens) / projectedTokensPerSec
	totalHours := totalSeconds / 3600
	totalCost := totalHours * target.OnDemandPerHr

	return fmt.Sprintf(
		"Projection: %dB model, %dB tokens on %s\n"+
			"  Estimated throughput: %.0f tokens/sec (%.0fx measured, 70%% efficiency)\n"+
			"  Estimated time: %.0f hours (%.1f days)\n"+
			"  Estimated cost: $%.0f (on-demand) / $%.0f (spot)\n"+
			"  Cost per 1M tokens: $%.2f",
		targetParams/1_000_000_000,
		targetTokens/1_000_000_000,
		targetInstanceType,
		projectedTokensPerSec,
		scaleFactor,
		totalHours,
		totalHours/24,
		totalCost,
		totalHours*target.SpotPerHr,
		(totalCost/float64(targetTokens))*1_000_000,
	)
}

func MFU(measuredTFLOPS, theoreticalTFLOPS float64) float64 {
	return math.Round(measuredTFLOPS/theoreticalTFLOPS*1000) / 10
}
