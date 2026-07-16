<script setup lang="ts">
import { onMounted, ref } from 'vue';
import { useRoute, useRouter, onBeforeRouteUpdate } from 'vue-router';
import { useI18n } from 'vue-i18n';
import { ApiService, Thread, HttpError, TestCaseResultStatus, TestSuiteRun, TestSuiteRunStatus } from '@/services/api';
import type { TestSuiteExecutionStreamEvent } from '@/services/api';
import AgentTestcasePanel from '@/components/agent/AgentTestcasePanel.vue';
import { useErrorHandler } from '@/composables/useErrorHandler';
import { useToast } from 'vue-toastification';
import ToastMessage from '@/components/common/ToastMessage.vue';
import { useTestCaseStore } from '@/composables/useTestCaseStore';
import { useTestExecutionStore } from '@/composables/useTestExecutionStore';
import { useToolAuthModal } from '@tero/common/utils/useToolAuthModal.js';
import { handleToolAuthRequestsIn } from '@/services/toolAuth';
import { AuthenticationError } from '@tero/common/utils/toolAuth.js';

export type { TestCaseExecutionState } from '@/composables/useTestExecutionStore';

const { t } = useI18n()
const { handleError } = useErrorHandler()
const toast = useToast()
const { testCasesStore, loadTestCases: loadTestCasesFromStore, clearTestCases } = useTestCaseStore()
const { testExecutionStore, loadSuiteRunResults, processStreamEvent, initializeTestRun, clear: clearExecutionStore } = useTestExecutionStore()
const api = new ApiService();
const route = useRoute();
const router = useRouter();
const agentId = ref<number>();
const threadId = ref<number>();
const showTestCaseEditor = ref<boolean>(false);
const isEditingTestCase = ref<boolean>(true);
const initialTab = ref<string>('0');
const testCasePanel = ref<InstanceType<typeof AgentTestcasePanel>>();
const loadingTests = ref<boolean>(true);
const testRunStartedByCurrentUser = ref<boolean>(false);
const isComparingResultWithTestSpec = ref<boolean>(false);

const { showToolAuthModal, toolAuthType, toolId, submitAuth, closeAuth } = useToolAuthModal()

const startChat = async () => {
  try {
    const thread = await api.startThread(agentId.value!);
    threadId.value = thread.id;
  } catch (e: unknown) {
    if (e instanceof HttpError && e.status === 400 && e.message.includes("Agent not found")) {
      router.push('/not-found');
    }
  }
}

const handleSelectChat = (chat: Thread) => {
  threadId.value = chat.id;
}

const handleSelectExecution = async (execution: TestSuiteRun) => {
  testExecutionStore.clear()
  testExecutionStore.setSelectedSuiteRun(execution)
  try {
    const results = await loadSuiteRunResults(agentId.value!, execution.id)
    testExecutionStore.setSelectedResult(results[0])
  } catch (error) {
    handleError(error)
  }
  isEditingTestCase.value = false
}

const processSuiteExecutionStream = async (
  eventStream: AsyncIterable<TestSuiteExecutionStreamEvent>,
  suiteRunId: number,
) => {
  let currentTestCaseResultId: number | undefined = undefined;
  try {
    for await (const event of eventStream) {
      const updatedResultId = processStreamEvent(event, {
        agentId: agentId.value,
        currentTestCaseResultId
      });
      currentTestCaseResultId = updatedResultId ?? currentTestCaseResultId;

      if (event.type === 'suite.error') {
        toast.error(
          { component: ToastMessage, props: { message: t('suiteExecutionFailed') } },
          { timeout: 5000, icon: false }
        );
      }
    }
  } catch (error) {
    if (currentTestCaseResultId) {
      const testCaseResult = testExecutionStore.testCaseResults.find(tr => tr.id === currentTestCaseResultId);
      if (testCaseResult) {
        testCaseResult.status = TestCaseResultStatus.ERROR;
      }
      testExecutionStore.setExecutionState(currentTestCaseResultId, {
        phase: 'completed',
        status: TestCaseResultStatus.ERROR
      });
    }
    handleError(error)
  } finally {
    if (testExecutionStore.selectedSuiteRun?.id === suiteRunId && testExecutionStore.selectedSuiteRun.status === TestSuiteRunStatus.RUNNING) {
      testExecutionStore.selectedSuiteRun.status = TestSuiteRunStatus.FAILURE;
      testExecutionStore.selectedSuiteRun.completedAt = new Date();
    }
    testExecutionStore.clearExecutionStates();
    testRunStartedByCurrentUser.value = false;
  }
}

const runTestSuite = async (testCaseIds?: number[]) => {
  isEditingTestCase.value = false
  testRunStartedByCurrentUser.value = true

  try {
    initializeTestRun(agentId.value!, testCasesStore.testCases, testCaseIds?.[0])
    const suiteRun = await handleToolAuthRequestsIn(
      () => api.runTestSuite(agentId.value!, testCaseIds),
      api
    )
    testExecutionStore.setSelectedSuiteRun(suiteRun)
    await processSuiteExecutionStream(api.streamTestSuiteUpdates(agentId.value!, suiteRun.id), suiteRun.id)
  } catch (error) {
    if (error instanceof AuthenticationError) {
      toast.error(
        { component: ToastMessage, props: { message: t(error.errorCode) } },
        { timeout: 5000, icon: false }
      )
      isEditingTestCase.value = true
      testRunStartedByCurrentUser.value = false
      return
    }
    await handleRunError(error)
  }
}

const handleRunError = async (error: unknown) => {
  if (error instanceof HttpError && error.status === 409) {
    try {
      toast.info(
        { component: ToastMessage, props: { message: t('suiteAlreadyRunning') } },
        { timeout: 5000, icon: false }
      )
      testRunStartedByCurrentUser.value = false
      const suiteRuns = await api.findTestSuiteRuns(agentId.value!, 1, 0)
      if (suiteRuns.length && suiteRuns[0].status === TestSuiteRunStatus.RUNNING) {
        testExecutionStore.setSelectedSuiteRun(suiteRuns[0])
        const results = await loadSuiteRunResults(agentId.value!, suiteRuns[0].id)
        testExecutionStore.setSelectedResult(results[0])
        await processSuiteExecutionStream(api.streamTestSuiteUpdates(agentId.value!, suiteRuns[0].id), suiteRuns[0].id)
      }
    } catch (error) {
      handleError(error)
    }
  } else {
    handleError(error)
    testRunStartedByCurrentUser.value = false
  }
}

const getTestcaseIdFromRoute = (routeObj: typeof route) => {
  const testcaseId = routeObj.query.testcaseId;
  return testcaseId ? parseInt(testcaseId as string) : undefined;
}

onMounted(async () => {
  agentId.value = parseInt(route.params.agentId as string);
  const testcaseId = getTestcaseIdFromRoute(route);
  if (testcaseId) {
    initialTab.value = '1';
  }
  await startChat();
  if (agentId.value) {
    await loadTestCases(agentId.value, testcaseId);
  }
  if (testcaseId) {
    showTestCaseEditor.value = true;
  }
});

const loadTestCases = async (id: number, testcaseId?: number) => {
  try {
    await loadTestCasesFromStore(id)

    if(testCasesStore.testCases.length) {
      if (testcaseId) {
        testCasesStore.setSelectedTestCaseById(testcaseId)
      }
      if (!testCasesStore.selectedTestCase) {
        testCasesStore.setSelectedTestCase(testCasesStore.testCases[0])
      }
    }

    const suiteRuns = await api.findTestSuiteRuns(id, 1, 0)
    if (suiteRuns.length && suiteRuns[0].status === TestSuiteRunStatus.RUNNING) {
      isEditingTestCase.value = false
      testExecutionStore.setSelectedSuiteRun(suiteRuns[0])
      const results = await loadSuiteRunResults(id, suiteRuns[0].id)
      testExecutionStore.setSelectedResult(results[0])
      processSuiteExecutionStream(api.streamTestSuiteUpdates(id, suiteRuns[0].id), suiteRuns[0].id)
    }
  } catch (e) {
    handleError(e)
  } finally {
    loadingTests.value = false
  }
}

const onEditingTestCase = (editing: boolean) => {
  if(testExecutionStore.selectedSuiteRun?.status == TestSuiteRunStatus.RUNNING) return
  isEditingTestCase.value = editing
}

onBeforeRouteUpdate(async (to) => {
  testRunStartedByCurrentUser.value = false;
  showTestCaseEditor.value = false;
  clearTestCases();
  clearExecutionStore();

  agentId.value = parseInt(to.params.agentId as string);
  const testcaseId = getTestcaseIdFromRoute(to);
  initialTab.value = testcaseId ? '1' : '0';
  await startChat();
  if (agentId.value) {
    await loadTestCases(agentId.value, testcaseId);
  }
  if (testcaseId) {
    showTestCaseEditor.value = true;
  }
});
</script>

<template>
  <PageLayout left-column-class="w-[calc(40%-var(--spacing)*3/2)] min-w-[670px]"
    right-column-class="w-[calc(60%-var(--spacing)*3/2)]">
    <template #left>
      <AgentEditorPanel v-if="threadId" :selected-thread-id="threadId"
        @show-test-case-editor="showTestCaseEditor = $event" :loading-tests="loadingTests"
        @editing-testcase="onEditingTestCase" :editing-testcase="isEditingTestCase"
        @import-agent="loadTestCases(agentId!)" :test-cases="testCasesStore.testCases" @run-tests="runTestSuite()"
        @run-single-test="(id: number) => runTestSuite([id])" @select-execution="handleSelectExecution"
        :is-comparing-result-with-test-spec="isComparingResultWithTestSpec"
        :test-spec-messages="testCasePanel?.testSpecMessages"
        :initial-tab="initialTab"/>
    </template>
    <template #right>
      <ChatPanel :thread-id="threadId" v-if="threadId && !showTestCaseEditor" :editing-agent="true"
        @new-chat="startChat" @select-chat="handleSelectChat" />
      <AgentTestcasePanel v-if="agentId && showTestCaseEditor" :agent-id="agentId" :thread-id="threadId"
        :is-editing="isEditingTestCase" @show-edit="isEditingTestCase = true"
        :is-new-test-case="testCasesStore.testCases.length === 0" ref="testCasePanel"
        @compare-result-with-test-spec-state-changed="isComparingResultWithTestSpec = $event" />
    </template>
  </PageLayout>
  <ToolAuthModal
    v-model:visible="showToolAuthModal"
    :tool-id="toolId"
    :auth-type="toolAuthType"
    @submit="submitAuth"
    @cancel="closeAuth"
  />
</template>

<i18n lang="json">
{
  "en": {
    "suiteExecutionFailed": "Test suite execution failed",
    "suiteAlreadyRunning": "Please wait for the test suite to finish running before starting a new execution",
    "authenticationWindowBlocked": "The authentication popup could not be opened. Please check that popups are allowed for this site and try again.",
    "authenticationCancelled": "The authentication was cancelled. Please, try again and complete the authentication to run the tests.",
    "authenticationAccessDenied": "The authentication was denied by the server. Please verify that you actually have the permissions necessary to use it."
  },
  "es": {
    "suiteExecutionFailed": "Falló la ejecución de la suite de tests",
    "suiteAlreadyRunning": "Espera a que el test suite termine de correr para lanzar una nueva ejecucion",
    "authenticationWindowBlocked": "No se pudo abrir la ventana de autenticación. Por favor, verifica que las ventanas emergentes estén permitidas para este sitio y vuelve a intentarlo.",
    "authenticationCancelled": "La autenticación fue cancelada. Por favor, inténtelo de nuevo y complete la autenticación para correr los tests.",
    "authenticationAccessDenied": "La autenticación fue denegada por el servidor. Por favor, verifica que tengas los permisos necesarios para usarlo."
  }
}
</i18n>
