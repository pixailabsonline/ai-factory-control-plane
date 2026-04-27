package orchestrator

import (
	"fmt"
	"os/exec"
	"strconv"
	"strings"
	"time"
)

type HealthStatus struct {
	NodeID    string
	Timestamp time.Time
	GPUs      []GPUHealth
	Healthy   bool
	Issues    []string
}

type GPUHealth struct {
	Index       int
	Name        string
	TempC       int
	UtilPct     int
	MemUsedMB   int
	MemTotalMB  int
	PCIeGen     int
	PCIeWidth   int
	NVLinkOK    bool
	Healthy     bool
	Issue       string
}

func CheckNodeHealth(nodeID string) (*HealthStatus, error) {
	status := &HealthStatus{
		NodeID:    nodeID,
		Timestamp: time.Now(),
		Healthy:   true,
	}

	gpus, err := queryNvidiaSMI()
	if err != nil {
		status.Healthy = false
		status.Issues = append(status.Issues, fmt.Sprintf("nvidia-smi failed: %v", err))
		return status, nil
	}

	for i := range gpus {
		gpu := &gpus[i]
		validateGPU(gpu)
		if !gpu.Healthy {
			status.Healthy = false
			status.Issues = append(status.Issues, fmt.Sprintf("GPU %d: %s", gpu.Index, gpu.Issue))
		}
	}

	status.GPUs = gpus
	return status, nil
}

func validateGPU(gpu *GPUHealth) {
	gpu.Healthy = true

	if gpu.TempC > 85 {
		gpu.Healthy = false
		gpu.Issue = fmt.Sprintf("temperature %d°C exceeds threshold (85°C)", gpu.TempC)
		return
	}

	if gpu.PCIeGen < 3 {
		gpu.Healthy = false
		gpu.Issue = fmt.Sprintf("PCIe Gen%d detected, expected Gen3+", gpu.PCIeGen)
		return
	}

	memUsedPct := float64(gpu.MemUsedMB) / float64(gpu.MemTotalMB) * 100
	if memUsedPct > 95 {
		gpu.Healthy = false
		gpu.Issue = fmt.Sprintf("memory %.1f%% used, likely OOM risk", memUsedPct)
		return
	}
}

func queryNvidiaSMI() ([]GPUHealth, error) {
	cmd := exec.Command("nvidia-smi",
		"--query-gpu=index,name,temperature.gpu,utilization.gpu,memory.used,memory.total,pcie.link.gen.current,pcie.link.width.current",
		"--format=csv,noheader,nounits",
	)

	out, err := cmd.Output()
	if err != nil {
		return nil, fmt.Errorf("nvidia-smi: %w", err)
	}

	var gpus []GPUHealth
	for _, line := range strings.Split(strings.TrimSpace(string(out)), "\n") {
		if line == "" {
			continue
		}
		fields := strings.Split(line, ", ")
		if len(fields) < 8 {
			continue
		}

		idx, _ := strconv.Atoi(strings.TrimSpace(fields[0]))
		temp, _ := strconv.Atoi(strings.TrimSpace(fields[2]))
		util, _ := strconv.Atoi(strings.TrimSpace(fields[3]))
		memUsed, _ := strconv.Atoi(strings.TrimSpace(fields[4]))
		memTotal, _ := strconv.Atoi(strings.TrimSpace(fields[5]))
		pcieGen, _ := strconv.Atoi(strings.TrimSpace(fields[6]))
		pcieWidth, _ := strconv.Atoi(strings.TrimSpace(fields[7]))

		gpus = append(gpus, GPUHealth{
			Index:      idx,
			Name:       strings.TrimSpace(fields[1]),
			TempC:      temp,
			UtilPct:    util,
			MemUsedMB:  memUsed,
			MemTotalMB: memTotal,
			PCIeGen:    pcieGen,
			PCIeWidth:  pcieWidth,
		})
	}

	return gpus, nil
}

func ValidatePCIeLinkSpeed(expectedGen int) error {
	gpus, err := queryNvidiaSMI()
	if err != nil {
		return err
	}
	for _, gpu := range gpus {
		if gpu.PCIeGen < expectedGen {
			return fmt.Errorf("GPU %d: PCIe Gen%d, expected Gen%d — degraded link", gpu.Index, gpu.PCIeGen, expectedGen)
		}
	}
	return nil
}
