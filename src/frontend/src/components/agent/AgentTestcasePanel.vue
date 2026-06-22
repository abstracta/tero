<script setup lang="ts">
import { TestCase, ThreadMessage, ThreadMessageOrigin, TestCaseResultStatus, TestSuiteRunStatus, TestCaseNewThreadMessage, TestCaseThreadMessageUpdate } from '@/services/api';
import { useI18n } from 'vue-i18n';
import { onMounted, onUnmounted, ref, watch, computed, nextTick } from 'vue';
import { ApiService, HttpError } from '@/services/api';
import { AgentTestcaseChatUiMessage } from './AgentTestcaseChatMessage.vue';
import AgentTestcaseInput from './AgentTestcaseInput.vue';
import { useErrorHandler } from '@/composables/useErrorHandler';
import { AnimationEffect } from '../../../../common/src/utils/animations';
import { initializeCodeActionHandler, renderMarkDown } from '../../../../common/src/utils/formatter';
import { useTestExecutionStore } from '@/composables/useTestExecutionStore';
import { useTestCaseStore } from '@/composables/useTestCaseStore';
import { IconArrowLeft, IconPencil, IconLoader2, IconSquareCheckFilled, IconSquareXFilled, IconSquareChevronsRightFilled } from '@tabler/icons-vue';

const api = new ApiService()
const { t } = useI18n()
const { handleError } = useErrorHandler()
const { testExecutionStore } = useTestExecutionStore()
const { testCasesStore, addTestCase, setSelectedTestCase, refreshTestCase } = useTestCaseStore()

const props = defineProps<{
    agentId: number,
    threadId?: number,
    isEditing?: boolean,
    isNewTestCase?: boolean
}>()

const emit = defineEmits<{
    (e: 'showEdit'): void
    (e: 'compareResultWithTestSpecStateChanged', isComparingResultWithTestSpec: boolean): void
}>()

const messages = ref<AgentTestcaseChatUiMessage[]>([])
const testSpecMessages = ref<AgentTestcaseChatUiMessage[]>([])
const inputText = ref('')
const selectedMessage = ref<AgentTestcaseChatUiMessage | undefined>()
const testcaseInputRef = ref<InstanceType<typeof AgentTestcaseInput>>()
const editingMessageUuid = ref<string | undefined>()
const editingMessageText = ref('')
const editingMessageOriginalText = ref('')
const editingInputRefs = ref<Map<string, InstanceType<typeof AgentTestcaseInput>>>(new Map())
const isLoading = ref(false)
const isComparingResultWithTestSpec = ref(false)
const showDeleteConfirmation = ref(false)
const messageToDelete = ref<AgentTestcaseChatUiMessage | undefined>()
const streamFlushIntervalId = ref<ReturnType<typeof setInterval> | null>(null)
const isRunning = computed(() => {
    if (executionState.value) {
        return executionState.value.phase === 'executing' || executionState.value.phase === 'evaluating'
    }
    return testCaseResult.value?.status === TestCaseResultStatus.RUNNING
})
const testCase = computed(() => testCasesStore.selectedTestCase)
const testCaseId = computed(() => testCase.value?.thread.id)
const testCaseResult = computed(() => { return testExecutionStore.selectedResult })
const testCaseResultTest = computed(() => {
    return testCaseResult.value ? testCasesStore.testCases.find(tc => tc.thread.id === testCaseResult.value!.testCaseId) : undefined
})
const executionState = computed(() => {
    return testCaseResult.value?.id ? testExecutionStore.getExecutionState(testCaseResult.value.id) : undefined
})

onMounted(async () => {
    initializeCodeActionHandler(t)
    isLoading.value = true
    try {
        if (testCaseId.value) {
            await loadTestCaseData()
        } else if (props.isNewTestCase) {
            messages.value = [
                new AgentTestcaseChatUiMessage(t('userMessagePlaceholder'), true, true),
                new AgentTestcaseChatUiMessage(t('agentMessagePlaceholder'), false, true),
            ]
            await selectNextMessageOrPlaceholder()
        }
    } finally {
        isLoading.value = false
    }
})

const loadTestCaseData = async () => {
    if (!props.isEditing && testCaseResult.value?.id && executionState.value?.phase === 'executing') {
        messages.value = []
        return
    }
    isLoading.value = true
    try {
        if (props.isEditing) {
            try {
                messages.value = mapMessagesToAgentTestcaseChatUi(await api.findTestCaseMessages(props.agentId, testCaseId.value!))
            } catch (error) {
                handleError(error)
            }
            await selectNextMessageOrPlaceholder()
        } else {
            try {
                if (testCaseResult.value?.testSuiteRunId && testCaseResult.value?.id) {
                    messages.value = mapMessagesToAgentTestcaseChatUi(
                        await api.findTestSuiteRunResultMessages(
                            props.agentId,
                            testCaseResult.value.testSuiteRunId,
                            testCaseResult.value.id
                        ),
                        false
                    )
                } else {
                    messages.value = []
                }
            } catch (error) {
                if (error instanceof HttpError && error.status === 404) {
                    messages.value = []
                } else {
                    handleError(error)
                }
            }
        }
    } finally {
        isLoading.value = false
    }
}

const mapMessagesToAgentTestcaseChatUi = (msgs: ThreadMessage[], usePlaceholders: boolean = true): AgentTestcaseChatUiMessage[] => {
    const result: AgentTestcaseChatUiMessage[] = msgs.map((msg, index) => {
        const isUser = index % 2 === 0
        const placeholderText = isUser ? t('userMessagePlaceholder') : t('agentMessagePlaceholder')
        const text = msg.text ?? (usePlaceholders ? placeholderText : '')

        return new AgentTestcaseChatUiMessage(
            text,
            isUser,
            usePlaceholders && msg.text == null,
            msg.id,
            msg.statusUpdates
        )
    })

    if (usePlaceholders) {
        if (result.length === 0) {
            result.push(new AgentTestcaseChatUiMessage(t('userMessagePlaceholder'), true, true))
            result.push(new AgentTestcaseChatUiMessage(t('agentMessagePlaceholder'), false, true))
        } else if (result[result.length - 1].isUser) {
            result.push(new AgentTestcaseChatUiMessage(t('agentMessagePlaceholder'), false, true))
        } else if (!result[result.length - 1].isPlaceholder) {
            result.push(new AgentTestcaseChatUiMessage(t('userMessagePlaceholder'), true, true))
            result.push(new AgentTestcaseChatUiMessage(t('agentMessagePlaceholder'), false, true))
        }
    }

    return result
}

const handleSelectMessage = async (message: AgentTestcaseChatUiMessage) => {
    if (message.isPlaceholder) {
        editingMessageUuid.value = undefined
        selectedMessage.value = message
        await nextTick()
        testcaseInputRef.value?.focus()
        inputText.value = ''
    } else {
        editingMessageUuid.value = message.uuid
        editingMessageText.value = message.text
        editingMessageOriginalText.value = message.text
        selectedMessage.value = undefined
        await nextTick()
        editingInputRefs.value.get(message.uuid)?.focus()
    }
}

const handleInlineEditSave = async (message: AgentTestcaseChatUiMessage) => {
    message.text = editingMessageText.value
    message.isPlaceholder = false
    editingMessageUuid.value = undefined
    editingMessageText.value = ''

    let currentTestCase = testCase.value;
    if (!currentTestCase) {
        currentTestCase = await addTestCase(props.agentId)
        setSelectedTestCase(currentTestCase)
    }
    saveMessageAsync(message, currentTestCase)

    const allMessagesSaved = messages.value.every(m => !m.isPlaceholder)
    if (allMessagesSaved) {
        const newUserPlaceholder = new AgentTestcaseChatUiMessage(
            t('userMessagePlaceholder'),
            true,
            true
        )
        const newAgentPlaceholder = new AgentTestcaseChatUiMessage(
            t('agentMessagePlaceholder'),
            false,
            true
        )
        messages.value.push(newUserPlaceholder)
        messages.value.push(newAgentPlaceholder)

        selectedMessage.value = newUserPlaceholder
        inputText.value = ''
        await nextTick()
        testcaseInputRef.value?.focus()
        return
    }

    if(getLastPlaceholder()) await selectNextMessageOrPlaceholder()
    else selectedMessage.value = undefined
}

const handleInlineEditCancel = async () => {
    const message = messages.value.find(m => m.uuid === editingMessageUuid.value)
    if (message) {
        message.text = editingMessageOriginalText.value
    }
    editingMessageUuid.value = undefined
    editingMessageText.value = ''
    editingMessageOriginalText.value = ''

    const lastPlaceholder = getLastPlaceholder()
    if (lastPlaceholder) {
        selectedMessage.value = lastPlaceholder
        inputText.value = ''
        await nextTick()
        testcaseInputRef.value?.focus()
    }
}

const handleSelectTestCase = async (newTestCase: TestCase | undefined) => {
    selectedMessage.value = undefined
    if (newTestCase) {
        if(!isRunning.value) await loadTestCaseData()
    } else if (props.isNewTestCase) {
        inputText.value = ''
        messages.value = [
            new AgentTestcaseChatUiMessage(t('userMessagePlaceholder'), true, true),
            new AgentTestcaseChatUiMessage(t('agentMessagePlaceholder'), false, true),
        ]
        await selectNextMessageOrPlaceholder()
    }
}

watch(() => testCase.value, async (newVal, oldVal) => {
    if ((newVal?.thread.id !== oldVal?.thread.id || props.isNewTestCase) && (testCasesStore.testCases.length > 1 || !oldVal)) {
        if (!oldVal && newVal && testCasesStore.testCases.length === 1) return
        await handleSelectTestCase(newVal)
    }
})

watch(() => testCaseResult.value, async () => {
    if (executionState.value?.phase === 'executing') {
        messages.value = []
        return
    }
    loadTestCaseData()
})

watch([() => props.isEditing, () => props.isNewTestCase], async () => {
    await handleSelectTestCase(testCase.value)
})

const flushStreamToMessage = () => {
    const state = executionState.value
    if (!state?.agentMessage) return
    const agentMsg = messages.value.slice().reverse().find((m: AgentTestcaseChatUiMessage) => !m.isUser)
    if (agentMsg && agentMsg.id === state.agentMessage!.id) {
        agentMsg.text = state.agentMessage!.text || ''
    }
}

onUnmounted(() => {
    if (streamFlushIntervalId.value != null) {
        clearInterval(streamFlushIntervalId.value)
        streamFlushIntervalId.value = null
    }
})

watch(executionState, (newState) => {
    if (!newState) {
        if (streamFlushIntervalId.value != null) {
            clearInterval(streamFlushIntervalId.value)
            streamFlushIntervalId.value = null
        }
        const lastAgentMsg = messages.value.slice().reverse().find((m: AgentTestcaseChatUiMessage) => !m.isUser)
        if (lastAgentMsg && (lastAgentMsg.isStreaming || !lastAgentMsg.isStatusComplete)) {
            lastAgentMsg.isStreaming = false
            lastAgentMsg.completeStatus()
        }
        return
    }

    if (newState.userMessage) {
        const userMessage = newState.userMessage
        const existingUserMsg = messages.value.find((m: AgentTestcaseChatUiMessage) => m.isUser && m.id === userMessage.id)
        if (!existingUserMsg) {
            messages.value.push(new AgentTestcaseChatUiMessage(
                userMessage.text || '',
                true,
                false,
                userMessage.id
            ))
        }
    }

    if (newState.agentMessage) {
        const agentText = newState.agentMessage.text || ''
        const isStreaming = newState.phase === 'executing' && !newState.agentMessage?.complete

        let agentMsg = messages.value.slice().reverse().find((m: AgentTestcaseChatUiMessage) => !m.isUser)

        if (!agentMsg || agentMsg.id !== newState.agentMessage.id) {
            agentMsg = new AgentTestcaseChatUiMessage(
                agentText,
                false,
                false,
                newState.agentMessage?.id
            )
            messages.value.push(agentMsg)
        } else if (!isStreaming) {
            agentMsg.text = agentText
        }

        agentMsg.id = newState.agentMessage.id
        agentMsg.isStreaming = isStreaming

        if (isStreaming && streamFlushIntervalId.value == null) {
           // Batch token updates every 100ms to avoid re-rendering on every incoming token
            streamFlushIntervalId.value = setInterval(flushStreamToMessage, 100)
        }
        if (!isStreaming && streamFlushIntervalId.value != null) {
            clearInterval(streamFlushIntervalId.value)
            streamFlushIntervalId.value = null
            agentMsg.text = agentText
        }

        if (newState.statusUpdates && newState.statusUpdates.length > 0) {
            agentMsg.statusUpdates = newState.statusUpdates.map((su: any) => ({
                action: su.action,
                toolName: su.toolName,
                description: su.description,
                args: su.args,
                step: su.step,
                result: su.result,
                timestamp: su.timestamp
            }))
        }

        if (newState.agentMessage?.complete === true) {
            agentMsg.completeStatus()
            agentMsg.isStreaming = false
        }
    }
}, { deep: true })

const onMessageSend = async () => {
    if (!selectedMessage.value) return

    const message = selectedMessage.value
    message.text = inputText.value
    message.isPlaceholder = false
    inputText.value = ''
    let currentTestCase = testCase.value;
    if (!currentTestCase) {
        currentTestCase = await addTestCase(props.agentId)
        setSelectedTestCase(currentTestCase)
    }
    saveMessageAsync(message, currentTestCase)

    const allMessagesSaved = messages.value.every(m => !m.isPlaceholder)
    if (allMessagesSaved) {
        const newUserPlaceholder = new AgentTestcaseChatUiMessage(
            t('userMessagePlaceholder'),
            true,
            true
        )
        const newAgentPlaceholder = new AgentTestcaseChatUiMessage(
            t('agentMessagePlaceholder'),
            false,
            true
        )
        messages.value.push(newUserPlaceholder)
        messages.value.push(newAgentPlaceholder)

        selectedMessage.value = newUserPlaceholder
        inputText.value = ''
        await nextTick()
        testcaseInputRef.value?.focus()
        return
    }

    if(getLastPlaceholder()) await selectNextMessageOrPlaceholder()
    else selectedMessage.value = undefined
}

const saveMessageAsync = (message: AgentTestcaseChatUiMessage, currentTestCase: TestCase) => {
    const apiCall = message.id
        ? api.updateTestCaseMessage(props.agentId, currentTestCase.thread.id, message.id, new TestCaseThreadMessageUpdate(message.text))
        : api.addTestCaseMessage(props.agentId, currentTestCase.thread.id,
            new TestCaseNewThreadMessage(message.text, message.isUser ? ThreadMessageOrigin.USER : ThreadMessageOrigin.AGENT))

    apiCall
        .then(async (updatedMessage: ThreadMessage) => {
            message.id = updatedMessage.id

            const firstUserMessage = messages.value[0]
            const firstAgentMessage = messages.value[1]
            const hasFirstUserMessage = firstUserMessage?.isUser && firstUserMessage?.text?.trim().length > 0 && !firstUserMessage.isPlaceholder
            const hasFirstAgentMessage = firstAgentMessage && !firstAgentMessage.isUser && firstAgentMessage?.text?.trim().length > 0 && !firstAgentMessage.isPlaceholder
            if (currentTestCase && currentTestCase.thread.id && hasFirstUserMessage && hasFirstAgentMessage) {
                await refreshTestCase(props.agentId, currentTestCase.thread.id)
            }
        })
        .catch((error: any) => {
            handleError(error)
        })
}

const selectNextMessageOrPlaceholder = async () => {
    if (selectedMessage.value) {
        const currentIndex = messages.value.findIndex(m => m.uuid === selectedMessage.value?.uuid)
        if (currentIndex !== -1 && currentIndex < messages.value.length - 1) {
            selectedMessage.value = messages.value[currentIndex + 1]
        }
    } else {
        selectedMessage.value = getLastPlaceholder()
    }
    await nextTick()
    inputText.value = selectedMessage.value?.isPlaceholder ? '' : selectedMessage.value?.text || ''
    if (selectedMessage.value) {
        testcaseInputRef.value?.focus()
    }
}

const getLastPlaceholder = () => {
    return messages.value.find(m => m.isPlaceholder)
}

const selectFirstPlaceholder = async () => {
    const firstPlaceholder = messages.value.find(m => m.isPlaceholder)
    if (firstPlaceholder) {
        selectedMessage.value = firstPlaceholder
        inputText.value = ''
        await nextTick()
        testcaseInputRef.value?.focus()
    }
}

const isMessageSelectable = (message: AgentTestcaseChatUiMessage) => {
    return getLastPlaceholder()?.uuid === message.uuid
}

const statusDescription = computed(() => {
    switch (testCaseResult.value?.status) {
        case TestCaseResultStatus.SUCCESS:
            return t('successDescription')
        case TestCaseResultStatus.FAILURE:
            return renderMarkDown(testCaseResult.value?.evaluatorAnalysis || t('failureDescription'), true)
        case TestCaseResultStatus.ERROR:
            if (testCaseResult.value?.errorCode === 'RECURSION_LIMIT_EXCEEDED') {
                return t('recursionLimitExceededDescription')
            }
            return t('errorDescription')
        default:
            return ''
    }
})

const toggleCompare = async () => {
    if (isComparingResultWithTestSpec.value) {
        isComparingResultWithTestSpec.value = false
        testSpecMessages.value = []
    } else {
        isComparingResultWithTestSpec.value = true
        await loadTestSpecMessages()
    }
    emit('compareResultWithTestSpecStateChanged', isComparingResultWithTestSpec.value)
}

const loadTestSpecMessages = async () => {
    try {
        const specMsgs = await api.findTestCaseMessages(props.agentId, testCaseResult.value!.testCaseId!)
        testSpecMessages.value = mapMessagesToAgentTestcaseChatUi(specMsgs, false)
    } catch (error) {
        handleError(error)
        testSpecMessages.value = []
    }
}

const showCompare = computed(() => {
    return testCaseResult.value && !isRunning.value && !props.isEditing && (testCaseResult.value?.status === TestCaseResultStatus.SUCCESS || testCaseResult.value?.status === TestCaseResultStatus.FAILURE)
})

const isCompareDisabled = computed(() => {
    return !existsTestCaseResultTest.value || isTestCaseResultTestModifiedAfterExecution.value
})

const compareButtonTooltip = computed(() => {
    if (isComparingResultWithTestSpec.value) {
        return t('closeCompare')
    }
    if (!existsTestCaseResultTest.value) {
        return t('compareDisabledReasonTestDeleted')
    }
    if (isTestCaseResultTestModifiedAfterExecution.value) {
        return t('compareDisabledReasonTestModified')
    }
    return t('compare')
})

const existsTestCaseResultTest = computed(() => {
    return testCaseResultTest.value !== undefined
})

const isTestCaseResultTestModifiedAfterExecution = computed(() => {
    return testCaseResultTest.value?.lastUpdate! > testCaseResult.value?.executedAt!
})

const handleDeleteMessageRequest = (message: AgentTestcaseChatUiMessage) => {
    messageToDelete.value = message
    showDeleteConfirmation.value = true
}

const cancelDeleteMessage = () => {
    showDeleteConfirmation.value = false
    messageToDelete.value = undefined
}

const confirmDeleteMessage = async () => {
    try {
        await api.deleteTestCaseMessageAndFollowing(props.agentId, testCaseId.value!, messageToDelete.value!.id!)
        await loadTestCaseData()
        if (testCaseId.value) {
            await refreshTestCase(props.agentId, testCaseId.value)
        }
        selectFirstPlaceholder()
    } catch (error) {
        handleError(error)
    } finally {
        cancelDeleteMessage()
    }
}

defineExpose({
    loadTestCaseData,
    isComparingResultWithTestSpec,
    testSpecMessages
})
</script>

<template>
    <Animate v-if="isLoading" :effect="AnimationEffect.FADE_IN" class="h-full">
        <AgentTestcasePanelSkeleton />
    </Animate>
    <FlexCard v-else header-height="auto">
        <template #header>
            <div class="flex flex-row items-center gap-4 h-10">
              <SimpleButton v-if="!isEditing && testCaseResult && testExecutionStore.selectedSuiteRun?.status !== TestSuiteRunStatus.RUNNING && !isComparingResultWithTestSpec" @click="$emit('showEdit')">
                  <IconArrowLeft />
              </SimpleButton>
              <div v-if="isEditing" class="flex flex-row gap-2">
                  <IconPencil v-if="testCase?.thread.name"/>
                  {{ testCase?.thread.name ? t('editTestCaseTitle', { testCaseName: testCase.thread.name }) :
                      t('newTestCaseTitle') }}
              </div>
              <div v-else class="flex gap-2 items-center">
                  <IconLoader2 v-if="isRunning" class="text-content-muted animate-spin" />
                  <div v-if="testCaseResult" class="flex flex-row items-center gap-2">
                    <SimpleIcon filled :icon="IconSquareCheckFilled" class="text-success" v-if="testCaseResult.status === TestCaseResultStatus.SUCCESS" v-tooltip.bottom="t('success')" />
                    <SimpleIcon filled :icon="IconSquareXFilled" class="text-error" v-if="testCaseResult.status === TestCaseResultStatus.FAILURE" v-tooltip.bottom="t('failure')" />
                    <IconExclamationMark class="text-white bg-warn rounded-sm p-0.5" size="20" v-if="testCaseResult.status === TestCaseResultStatus.ERROR" v-tooltip.bottom="t('error')" />
                    <SimpleIcon filled :icon="IconSquareChevronsRightFilled" class="text-content-muted" v-if="testCaseResult.status === TestCaseResultStatus.SKIPPED" v-tooltip.bottom="t('skipped')" />
                    {{ testCaseResult ? isRunning ? t('runningTestCase', { testCaseName: testCaseResult.testCaseName }) : t('testCaseResult', { testCaseName: testCaseResult.testCaseName }) : t('noRunResults') }}
                </div>
              </div>
              <div v-if="showCompare" class="ml-auto">
                  <SimpleButton @click="toggleCompare" :disabled="isCompareDisabled" v-tooltip.bottom="compareButtonTooltip">
                      <IconColumns v-if="!isComparingResultWithTestSpec"/>
                      <IconX v-else/>
                  </SimpleButton>
              </div>
            </div>
        </template>
        <div class="max-w-[837px] mx-auto flex-1 w-full min-h-0">
            <div class="flex flex-col h-full gap-4 py-4">
                <div class="flex flex-1 min-h-0 gap-2 w-full overflow-y-auto">
                    <div v-if="messages.length > 0 && (isEditing || testCaseResult?.status !== TestCaseResultStatus.PENDING)" class="flex flex-col w-full pt-6">
                        <div v-for="message in messages" :key="message.uuid">
                            <div v-if="!message.isUser && message.statusUpdates.length > 0 && !isComparingResultWithTestSpec" class="px-2 py-3">
                                <ChatStatus :status-updates="message.statusUpdates" :is-complete="message.isStatusComplete" />
                            </div>
                            <AgentTestcaseChatMessage
                                v-if="editingMessageUuid !== message.uuid"
                                :message="message"
                                :actions-enabled="!message.isPlaceholder && isEditing"
                                :is-selected="selectedMessage?.uuid === message.uuid"
                                @select="handleSelectMessage"
                                @delete="handleDeleteMessageRequest"
                                :selectable="isMessageSelectable(message)" />
                            <AgentTestcaseInput
                                v-else
                                :ref="(el) => { if (el) editingInputRefs.set(message.uuid, el as InstanceType<typeof AgentTestcaseInput>) }"
                                :selected-message="message"
                                v-model="editingMessageText"
                                :handle-error="handleError"
                                :is-editing="true"
                                :agent-id="agentId"
                                @save="handleInlineEditSave(message)"
                                @cancel="handleInlineEditCancel" />
                        </div>
                    </div>
                </div>
                <AgentTestcaseInput
                    v-if="isEditing && selectedMessage"
                    ref="testcaseInputRef"
                    :selected-message="selectedMessage"
                    v-model="inputText"
                    :handle-error="handleError"
                    :agent-id="agentId"
                    @send="onMessageSend"
                    :enable-prompts="true" />
                <div v-else-if="!isEditing && (testCaseResult || executionState)"
                    class="h-40 min-h-40 flex-shrink-0 border-t-1 p-4">
                    <div class="flex flex-col gap-4 h-full">
                        <AgentTestcaseStatus v-if="!isRunning && testCaseResult" :status="testCaseResult.status" />
                        <div class="h-full overflow-y-auto">
                            <div v-if="executionState && (executionState.phase === 'executing' || executionState.phase === 'evaluating')" class="flex flex-row items-center gap-2">
                                <IconLoader2 class="text-content-muted animate-spin" />
                                <span v-if="executionState.phase === 'executing'">{{ t('phaseExecuting') }}</span>
                                <span v-else>{{ t('phaseEvaluating') }}</span>
                            </div>
                            <div v-else-if="isRunning" class="flex flex-row items-center gap-2">
                                <IconLoader2 class="text-content-muted animate-spin" />
                                <span>{{ t('testRunning') }}</span>
                            </div>
                            <div v-else-if="testCaseResult" class="flex flex-col gap-2" v-html="statusDescription" />
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </FlexCard>
    <Dialog v-model:visible="showDeleteConfirmation" :header="t('deleteConfirmTitle')" :modal="true" :draggable="false"
        :resizable="false" :closable="false" class="max-w-150">
        <div class="flex flex-col gap-5">
            <div class="flex flex-row gap-2 items-start whitespace-pre-line">
                {{ t('deleteConfirmDescription') }}
            </div>
            <div class="flex flex-row gap-2 justify-end">
                <SimpleButton @click="cancelDeleteMessage" shape="square" variant="secondary">{{ t('cancelDeleteButton') }}</SimpleButton>
                <SimpleButton @click="confirmDeleteMessage" variant="error" shape="square">{{ t('deleteButton') }}</SimpleButton>
            </div>
        </div>
    </Dialog>
</template>

<i18n lang="json">
{
    "en": {
        "newTestCaseTitle": "Create a new test case",
        "editTestCaseTitle": "Editing: {testCaseName}",
        "userMessagePlaceholder": "Message that a person would send to the agent…",
        "agentMessagePlaceholder": "Expected response from the agent…",
        "user": "Message from the user",
        "agent": "Expected response",
        "noRunResults": "No execution results yet",
        "successDescription": "The agent's response matched the expected output. No formatting or content deviations were detected.",
        "failureDescription": "The agent's response did not match the expected output. Formatting or content deviations were detected.",
        "errorDescription": "An error occurred while running the test case",
        "recursionLimitExceededDescription": "The step limit for this response was reached. Try a shorter task or break your request into smaller parts. You can review the thought process to improve the use of the agent and avoid steps that you identify as unnecessary. To increase the limit, go to the agent settings and adjust the thought process steps limit.",
        "runningTestCase": "Running: {testCaseName}",
        "testCaseResult": "Test case result: {testCaseName}",
        "testRunning": "Test is running...",
        "phaseExecuting": "Executing test case...",
        "phaseEvaluating": "Evaluating response...",
        "compare": "Compare with specification",
        "closeCompare": "Close compare",
        "compareDisabledReasonTestModified": "Compare is disabled because the test case was modified after the test execution",
        "compareDisabledReasonTestDeleted": "Compare is disabled because the test case was deleted",
        "success": "Success",
        "failure": "Failed",
        "error": "Error running",
        "skipped": "Skipped",
        "deleteConfirmTitle": "Delete message",
        "deleteConfirmDescription": "Are you sure you want to delete this message and all following messages? This action cannot be undone.",
        "cancelDeleteButton": "Cancel",
        "deleteButton": "Delete"
    },
    "es": {
        "newTestCaseTitle": "Crea un nuevo test case",
        "editTestCaseTitle": "Editando: {testCaseName}",
        "userMessagePlaceholder": "Mensaje que enviaría una persona al agente…",
        "agentMessagePlaceholder": "Respuesta esperada del agente…",
        "user": "Mensaje del usuario",
        "agent": "Respuesta esperada",
        "noRunResults": "No hay resultados de ejecución aún",
        "successDescription": "La respuesta del agente coincidió con la salida esperada. No se detectaron desvíos de formato o contenido.",
        "failureDescription": "La respuesta del agente no coincidió con la salida esperada. Se detectaron desvíos de formato o contenido.",
        "errorDescription": "Ocurrió un error al ejecutar el test case",
        "recursionLimitExceededDescription": "Se alcanzó el límite de pasos de esta respuesta. Intenta con una tarea más corta o divide la solicitud en partes más pequeñas. Puedes revisar el proceso de pensamiento para mejorar el uso del agente y evitar pasos que identifiques que no sean necesarios. Para aumentar el límite, ve a la configuración del agente y ajusta el límite de pasos del proceso de pensamiento.",
        "runningTestCase": "Ejecutando: {testCaseName}",
        "testCaseResult": "Resultado del test case: {testCaseName}",
        "testRunning": "El test está ejecutándose...",
        "phaseExecuting": "Ejecutando test case...",
        "phaseEvaluating": "Evaluando respuesta...",
        "compare": "Comparar con la especificación",
        "closeCompare": "Cerrar comparación",
        "compareDisabledReasonTestModified": "La comparación está deshabilitada porque el test case fue modificado después de la ejecución del test",
        "compareDisabledReasonTestDeleted": "La comparación está deshabilitada porque el test case fue eliminado",
        "success": "Pasó",
        "failure": "Falló",
        "error": "Error al ejecutar",
        "skipped": "Omitido",
        "deleteConfirmTitle": "Eliminar mensaje",
        "deleteConfirmDescription": "¿Estás seguro de que deseas eliminar este mensaje y todos los siguientes? Esta acción no se puede deshacer.",
        "cancelDeleteButton": "Cancelar",
        "deleteButton": "Eliminar"
    }
}
</i18n>
