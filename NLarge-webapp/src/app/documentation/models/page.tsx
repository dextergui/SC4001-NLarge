import { CodeHighlightTabs } from "@mantine/code-highlight";
import {
  AppShellSection,
  Code,
  Divider,
  Group,
  rem,
  Stack,
  Text,
  Title,
} from "@mantine/core";

export default function DocumentationModel() {
  return (
    <>
      <AppShellSection className="pt-10" bg="bg2">
        <Group justify="center">
          <Stack className="w-3/4 p-2 pb-10">
            <Title
              c="primary"
              ff="monospace"
              size={rem(42)}
              fw="bolder"
              mt="xl"
              ta="left"
            >
              Language Models
            </Title>
            <Text c="dimmed" size="lg">
              For your convenience, we have included some ready-to-use Neural
              Network Models in NLarge.
            </Text>
          </Stack>
        </Group>
        <Divider my="lg" />
      </AppShellSection>
      <AppShellSection className="pb-10">
        <Group justify="center">
          <Group className="w-2/3">
            <Stack>
              <Text c="primary" size="lg" fw="bolder">
                Ready-to-Use Models
              </Text>
              <Text c="dimmed" size="md">
                Some NLP models are implemented as classes for the convenience
                of use. Each class is implemented to provide users with
                different neural network architectures for text classification,
                allowing flexibility in model selection based on specific needs
                like attention mechanisms, multi-head attention, or recurrent
                layers such as LSTM, GRU, and vanilla RNN.
              </Text>
              <Divider my={2} />

              <Text c="primary" size="lg" fw="bolder">
                <Code bg="dimmed" fz="lg">
                  RNN.py
                </Code>
              </Text>
              <Text c="dimmed" size="md">
                This module offers models based on a vanilla RNN for text
                classification, providing alternatives with and without max
                pooling.
              </Text>

              <Text c="primary" size="md" fw="bolder">
                1. TextClassifierRNN
              </Text>
              <Text c="dimmed" size="md">
                A simple RNN-based classifier, utilizing a fully connected layer
                after the final hidden state for sequence classification.
                Suitable for shorter sequences where RNN limitations, such as
                gradient vanishing, are less significant.
              </Text>

              <Text c="primary" size="md" fw="bolder">
                2. TextClassifierRNNMaxPool
              </Text>
              <Text c="dimmed" size="md">
                Extends <Code bg="dimmed">TextClassifierRNN</Code> by applying a
                max-pooling operation over the RNN outputs. This version can
                better capture the most relevant features across the entire
                sequence.
              </Text>

              <Text c="primary" size="md" fw="bolder">
                Example Usage:
              </Text>
              <CodeHighlightTabs
                code={[
                  {
                    fileName: "python",
                    code: `
# Import pipeline and Model
from NLarge.pipeline import TextClassificationPipeline
from NLarge.model.RNN import TextClassifierRNN 

# Initialize Pipeline
pipeline_augmented = TextClassificationPipeline(
    augmented_data=augmented_train_data,
    test_data=original_test_data,
    max_length=128,
    test_size=0.2,
    model_class=TextClassifierRNN,
)
                    `,
                    language: "python",
                  },
                ]}
                className="w-full rounded-sm outline-1 outline outline-slate-600"
              />

              <Divider my={2} />

              <Text c="primary" size="lg" fw="bolder">
                <Code bg="dimmed" fz="lg">
                  LSTM.py
                </Code>
              </Text>
              <Text c="dimmed" size="md">
                This module provides LSTM-based models for text classification,
                with and without attention mechanisms.
              </Text>

              <Text c="primary" size="md" fw="bolder">
                1. TextClassifierLSTM
              </Text>
              <Text c="dimmed" size="md">
                Implements a bidirectional LSTM classifier, where the final
                hidden state of the sequence is passed to a fully connected
                layer for binary classification.
              </Text>

              <Text c="primary" size="md" fw="bolder">
                2. Attention
              </Text>
              <Text c="dimmed" size="md">
                An attention layer for use with LSTM outputs. It computes
                attention scores for each step in the sequence, providing a
                weighted sum of hidden states based on their importance.
              </Text>

              <Text c="primary" size="md" fw="bolder">
                3. TextClassifierLSTMWithAttention
              </Text>
              <Text c="dimmed" size="md">
                An LSTM classifier incorporating the attention mechanism. It
                uses a bidirectional LSTM to capture contextual information,
                followed by attention to highlight critical sequence components,
                which enhances interpretability.
              </Text>
              <Text c="primary" size="md" fw="bolder">
                Example Usage:
              </Text>
              <CodeHighlightTabs
                code={[
                  {
                    fileName: "python",
                    code: `
# Import pipeline and Model
from NLarge.pipeline import TextClassificationPipeline
from NLarge.model.LSTM import TextClassifierLSTMWithAttention 

# Initialize Pipeline
pipeline_augmented = TextClassificationPipeline(
    augmented_data=augmented_train_data,
    test_data=original_test_data,
    max_length=128,
    test_size=0.2,
    model_class=TextClassifierLSTMWithAttention,
)
                    `,
                    language: "python",
                  },
                ]}
                className="w-full rounded-sm outline-1 outline outline-slate-600"
              />

              <Divider my={2} />

              <Text c="primary" size="lg" fw="bolder">
                <Code bg="dimmed" fz="lg">
                  GRU.py
                </Code>
              </Text>
              <Text c="dimmed" size="md">
                This module contains GRU-based models for text classification,
                providing options with and without attention.
              </Text>

              <Text c="primary" size="md" fw="bolder">
                1. TextClassifierGRU
              </Text>
              <Text c="dimmed" size="md">
                Implements a GRU-based classifier using bidirectional GRU
                layers, allowing context from both directions in the sequence.
                It applies a fully connected layer on the final hidden state,
                suitable for capturing sequential information.
              </Text>

              <Text c="primary" size="md" fw="bolder">
                2. Attention
              </Text>
              <Text c="dimmed" size="md">
                This class defines an attention mechanism specific to the GRU
                output, helping focus on the most informative parts of the
                sequence.
              </Text>

              <Text c="primary" size="md" fw="bolder">
                3. TextClassifierGRUWithAttention
              </Text>
              <Text c="dimmed" size="md">
                A GRU-based classifier that incorporates attention to focus on
                important sequence parts before final classification. This model
                uses a bidirectional GRU, with attention applied to the output,
                and aggregates the context vector to the final output layer.
              </Text>
              <Text c="primary" size="md" fw="bolder">
                Example Usage:
              </Text>
              <CodeHighlightTabs
                code={[
                  {
                    fileName: "python",
                    code: `
# Import pipeline and Model
from NLarge.pipeline import TextClassificationPipeline
from NLarge.model.GRU import TextClassifierGRU 

# Initialize Pipeline
pipeline_augmented = TextClassificationPipeline(
    augmented_data=augmented_train_data,
    test_data=original_test_data,
    max_length=128,
    test_size=0.2,
    model_class=TextClassifierGRU,
)
                    `,
                    language: "python",
                  },
                ]}
                className="w-full rounded-sm outline-1 outline outline-slate-600"
              />

              <Divider my={2} />

              <Text c="primary" size="lg" fw="bolder">
                <Code bg="dimmed" fz="lg">
                  attention.py
                </Code>
              </Text>
              <Text c="dimmed" size="md">
                This module provides attention-based models for text
                classification.
              </Text>

              <Text c="primary" size="md" fw="bolder">
                1. TextClassifierAttentionNetwork
              </Text>
              <Text c="dimmed" size="md">
                Implements a basic attention mechanism with query, key, and
                value layers. The model calculates attention weights, aggregates
                them to form a context vector, and then applies a fully
                connected layer followed by a sigmoid activation for binary
                classification.
              </Text>

              <Text c="primary" size="md" fw="bolder">
                2. MultiHeadAttention
              </Text>
              <Text c="dimmed" size="md">
                A multi-head attention layer that splits the input into multiple
                heads to capture various aspects of the input representation,
                making it effective in handling complex relationships in the
                input sequence.
              </Text>

              <Text c="primary" size="md" fw="bolder">
                3. TextClassifierMultiHeadAttentionNetwork
              </Text>
              <Text c="dimmed" size="md">
                Uses the <Code bg="dimmed">MultiHeadAttention</Code> class for
                multi-head attention followed by a fully connected layer for
                classification. It aggregates attention across multiple heads to
                increase model interpretability and capture richer
                sequence-level information.
              </Text>
              <Text c="primary" size="md" fw="bolder">
                Example Usage:
              </Text>
              <CodeHighlightTabs
                code={[
                  {
                    fileName: "python",
                    code: `
# Import pipeline and Model
from NLarge.pipeline import TextClassificationPipeline
from NLarge.model.Attention import MultiHeadAttention 

# Initialize Pipeline
pipeline_augmented = TextClassificationPipeline(
    augmented_data=augmented_train_data,
    test_data=original_test_data,
    max_length=128,
    test_size=0.2,
    model_class=MultiHeadAttention,
)
                    `,
                    language: "python",
                  },
                ]}
                className="w-full rounded-sm outline-1 outline outline-slate-600"
              />

              <Divider my={2} />
            </Stack>
          </Group>
        </Group>
      </AppShellSection>
    </>
  );
}
