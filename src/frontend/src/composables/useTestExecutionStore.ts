import { reactive } from 'vue'
import { ApiService, TestCase, TestCaseResult, TestCaseResultStatus, TestSuiteRun, TestSuiteRunStatus } from '@/services/api'
import type { TestSuiteExecutionStreamEvent } from '@/services/api'

export interface TestCaseExecutionState {
    phase: string;
    userMessage?: { id: number; text: string };
    agentMessage?: { id: number; text: string, complete: boolean };
    status?: TestCaseResultStatus;
    statusUpdates?: any[];
}

const testExecutionStore = reactive({
    selectedSuiteRun: undefined as TestSuiteRun | undefined,
    selectedResult: undefined as TestCaseResult | undefined,
    testCaseResults: [] as TestCaseResult[],
    executionStates: new Map<number, TestCaseExecutionState>(),
    isStoppingSuite: false,
    streamingSuiteRunId: undefined as number | undefined,

    setSelectedSuiteRun(suiteRun: TestSuiteRun | undefined) {
        this.selectedSuiteRun = suiteRun
    },

    setSelectedResult(result: TestCaseResult | undefined) {
        this.selectedResult = result
    },

    setTestCaseResults(results: TestCaseResult[]) {
        this.testCaseResults = results
        this.selectedResult = results.find(r => r.testCaseId === this.selectedResult?.testCaseId)
    },

    setExecutionState(testCaseResultId: number, state: TestCaseExecutionState) {
        this.executionStates.set(testCaseResultId, state)
    },

    getExecutionState(testCaseResultId: number): TestCaseExecutionState | undefined {
        return this.executionStates.get(testCaseResultId)
    },

    deleteExecutionState(testCaseResultId: number) {
        this.executionStates.delete(testCaseResultId)
    },

    clearExecutionStates() {
        this.executionStates.clear()
    },

    setTestCaseResultStatus(testCaseId: number, status: TestCaseResultStatus) {
        const result = this.testCaseResults.find(tr => tr.testCaseId === testCaseId)
        if (!result) {
            return
        }

        result.status = status

        if (this.selectedResult?.testCaseId === testCaseId) {
            this.selectedResult.status = status
        }
    },

    clear() {
        this.selectedSuiteRun = undefined
        this.selectedResult = undefined
        this.testCaseResults = []
        this.executionStates.clear()
    }
})

export function useTestExecutionStore() {
    const api = new ApiService()

    async function loadSuiteRunResults(agentId: number, suiteRunId: number) {
        const results = await api.findTestSuiteRunResults(agentId, suiteRunId)
        testExecutionStore.setTestCaseResults(results)
        return results
    }

    function processStreamEvent(
        event: TestSuiteExecutionStreamEvent,
        options?: {
            currentTestCaseResultId?: number
            agentId?: number
        }
    ): number | undefined {
        let currentTestCaseResultId: number | undefined = options?.currentTestCaseResultId

        switch (event.type) {
            case 'suite.test.start':
                const testCaseId = event.data.testCaseId;
                currentTestCaseResultId = event.data.resultId;

                const testCaseResult = testExecutionStore.testCaseResults.find(tr => tr.testCaseId === testCaseId);
                if (testCaseResult) {
                    testExecutionStore.setTestCaseResultStatus(testCaseId, TestCaseResultStatus.RUNNING);
                    testCaseResult.id = currentTestCaseResultId;
                }

                testExecutionStore.setExecutionState(currentTestCaseResultId, {
                    phase: 'executing',
                    statusUpdates: [{ action: 'statusProcessing', timestamp: new Date() }]
                });

                break;

            case 'suite.test.metadata':
                const testResult = testExecutionStore.testCaseResults.find(tr => tr.testCaseId === event.data.testCaseId);
                if (testResult) {
                    testResult.id = event.data.resultId;
                    if (testExecutionStore.selectedSuiteRun) {
                        testResult.testSuiteRunId = testExecutionStore.selectedSuiteRun.id;
                    }
                }
                break;

            case 'suite.test.phase':
                const execState = testExecutionStore.getExecutionState(currentTestCaseResultId!);
                if (execState) {
                    execState.phase = event.data.phase;
                    if (event.data.phase === 'completed') {
                        execState.status = event.data.status as TestCaseResultStatus;
                    }
                }
                break;

            case 'suite.test.userMessage':
                const userExecState = testExecutionStore.getExecutionState(currentTestCaseResultId!);
                if (userExecState) {
                    userExecState.userMessage = {
                        id: event.data.id,
                        text: event.data.text
                    };
                }
                break;

            case 'suite.test.agentMessage.start':
                const startExecState = testExecutionStore.getExecutionState(currentTestCaseResultId!);
                if (startExecState) {
                    startExecState.agentMessage = {
                        id: event.data.id,
                        text: '',
                        complete: false
                    };
                }
                break;

            case 'suite.test.agentMessage.chunk':
                const chunkExecState = testExecutionStore.getExecutionState(currentTestCaseResultId!);
                if (chunkExecState && chunkExecState.agentMessage) {
                    chunkExecState.agentMessage.text += event.data.chunk;
                }
                break;

            case 'suite.test.agentMessage.complete':
                const completeExecState = testExecutionStore.getExecutionState(currentTestCaseResultId!);
                if (completeExecState && completeExecState.agentMessage) {
                    completeExecState.agentMessage.text = event.data.text;
                    completeExecState.agentMessage.complete = true;
                    const lastStatus = completeExecState.statusUpdates?.slice(-1)[0];
                    if (lastStatus) lastStatus.timestamp = new Date();
                }
                break;

            case 'suite.test.executionStatus':
                const statusExecState = testExecutionStore.getExecutionState(currentTestCaseResultId!);
                if (statusExecState) {
                    statusExecState.statusUpdates!.push({ ...event.data, timestamp: new Date() });
                }
                break;

            case 'suite.test.error':
                const errorExecState = testExecutionStore.getExecutionState(currentTestCaseResultId!);
                if (errorExecState) {
                    errorExecState.phase = 'completed';
                    errorExecState.status = TestCaseResultStatus.ERROR;
                }

                const errorResult = testExecutionStore.testCaseResults.find(tr => tr.id === currentTestCaseResultId!);
                if (errorResult) {
                    testExecutionStore.setTestCaseResultStatus(errorResult.testCaseId, TestCaseResultStatus.ERROR);
                }
                break;

            case 'suite.test.complete':
                const completedTestCaseId = event.data.testCaseId;
                updateTestCaseResult(completedTestCaseId, event.data.status as TestCaseResultStatus, event.data.evaluation?.analysis, event.data.errorCode);

                const completedResult = testExecutionStore.testCaseResults.find(tr => tr.testCaseId === completedTestCaseId);
                if (completedResult?.id) {
                    testExecutionStore.deleteExecutionState(completedResult.id);
                }
                break;

            case 'suite.error':
                testExecutionStore.testCaseResults.forEach(result => {
                    if (result.status === TestCaseResultStatus.RUNNING || result.status === TestCaseResultStatus.PENDING) {
                        testExecutionStore.setTestCaseResultStatus(result.testCaseId, TestCaseResultStatus.SKIPPED);
                    }
                });

                if (testExecutionStore.selectedSuiteRun) {
                    testExecutionStore.selectedSuiteRun.status = TestSuiteRunStatus.FAILURE
                    testExecutionStore.selectedSuiteRun.completedAt = new Date()
                }

                testExecutionStore.clearExecutionStates();
                break;

            case 'suite.complete':
                if (testExecutionStore.selectedSuiteRun) {
                    testExecutionStore.selectedSuiteRun.status = event.data.status as TestSuiteRunStatus
                    testExecutionStore.selectedSuiteRun.completedAt = new Date()
                    testExecutionStore.selectedSuiteRun.passedTests = event.data.passed
                    testExecutionStore.selectedSuiteRun.failedTests = event.data.failed
                    testExecutionStore.selectedSuiteRun.errorTests = event.data.errors
                    testExecutionStore.selectedSuiteRun.skippedTests = event.data.skipped
                }
                break;
        }

        return currentTestCaseResultId;
    }

    function updateTestCaseResult(testCaseId: number, status: TestCaseResultStatus, evaluatorAnalysis?: string, errorCode?: string) {
        const result = testExecutionStore.testCaseResults.find(tr => tr.testCaseId === testCaseId)
        if (result) {
            result.status = status
            result.evaluatorAnalysis = evaluatorAnalysis
            result.errorCode = errorCode
            if (testExecutionStore.selectedResult?.testCaseId === testCaseId) {
                testExecutionStore.selectedResult = result
            }
        }
    }

    function initializeTestRun(agentId: number, testCases: TestCase[], singleTestCaseId?: number) {
        const results = testCases.map(testCase => {
            return new TestCaseResult(
                testCase.thread.id,
                new Date(),
                singleTestCaseId ? (testCase.thread.id === singleTestCaseId ? TestCaseResultStatus.PENDING : TestCaseResultStatus.SKIPPED) : TestCaseResultStatus.PENDING,
                testCase.thread.name
            )
        })
        testExecutionStore.setTestCaseResults(results)
        setSelectedResult(singleTestCaseId ? results.find(result => result.testCaseId === singleTestCaseId)! : results[0]);

        testExecutionStore.setSelectedSuiteRun({
            id: 0,
            agentId: agentId,
            status: TestSuiteRunStatus.RUNNING,
            executedAt: new Date(),
            totalTests: testCases.length,
            passedTests: 0,
            failedTests: 0,
            errorTests: 0,
            skippedTests: 0
        })
    }

    function setSelectedResult(result: TestCaseResult) {
        testExecutionStore.setSelectedResult(result)
    }

    async function stopSuiteRun() {
        const selectedSuiteRun = testExecutionStore.selectedSuiteRun
        if (!selectedSuiteRun || testExecutionStore.isStoppingSuite) {
            return
        }

        testExecutionStore.isStoppingSuite = true

        try {
            await api.stopTestSuiteRun(selectedSuiteRun.agentId, selectedSuiteRun.id)
            selectedSuiteRun.status = TestSuiteRunStatus.CANCELLING
            testExecutionStore.testCaseResults.forEach(result => {
                if (result.status === TestCaseResultStatus.RUNNING || result.status === TestCaseResultStatus.PENDING) {
                    testExecutionStore.setTestCaseResultStatus(result.testCaseId, TestCaseResultStatus.SKIPPED)
                }
            })
            testExecutionStore.clearExecutionStates()
        } finally {
            testExecutionStore.isStoppingSuite = false
        }
    }

    function clear() {
        testExecutionStore.streamingSuiteRunId = undefined
        testExecutionStore.clear()
    }

    return {
        testExecutionStore,
        loadSuiteRunResults,
        processStreamEvent,
        updateTestCaseResult,
        initializeTestRun,
        stopSuiteRun,
        setSelectedResult,
        clear,
    }
}
