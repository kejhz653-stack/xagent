"""Tests for vector_manager functionality.

This module tests the vector storage data management functions:
- read_chunks_for_embedding: Reading chunks from database for embedding
- write_vectors_to_db: Writing embedding vectors with idempotency
- validate_query_vector: Vector validation functionality
- Vector consistency and error handling
"""

import os
import tempfile
import uuid
from unittest.mock import MagicMock, patch

import pytest

from xagent.core.tools.core.RAG_tools.core.exceptions import VectorValidationError
from xagent.core.tools.core.RAG_tools.core.schemas import (
    EmbeddingReadResponse,
    EmbeddingWriteResponse,
)
from xagent.core.tools.core.RAG_tools.vector_storage.vector_manager import (
    _group_embeddings_by_model,
    _validate_and_prepare_table,
    read_chunks_for_embedding,
    validate_query_vector,
    write_vectors_to_db,
)


def _create_mock_table_with_schema() -> MagicMock:
    """Create a mock table with a schema that includes the metadata field.

    This helper function ensures that schema validation passes in tests
    by providing a mock schema that includes all required fields, especially
    the 'metadata' field that is validated in ensure_chunks_table and
    ensure_embeddings_table.

    Returns:
        A MagicMock table object with a properly configured schema.
    """
    table = MagicMock()
    # Create mock schema fields - at minimum include 'metadata' which is validated
    metadata_field = MagicMock()
    metadata_field.name = "metadata"
    collection_field = MagicMock()
    collection_field.name = "collection"
    doc_id_field = MagicMock()
    doc_id_field.name = "doc_id"
    # Set schema as a list of field objects (mimicking PyArrow schema structure)
    table.schema = [collection_field, doc_id_field, metadata_field]
    return table


class TestReadChunksForEmbedding:
    """Test cases for read_chunks_for_embedding functionality."""

    @pytest.fixture
    def temp_lancedb_dir(self):
        """Create a temporary directory for LanceDB."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Set the environment variable for LanceDB directory
            original_env = os.environ.get("LANCEDB_DIR")
            os.environ["LANCEDB_DIR"] = temp_dir
            yield temp_dir
            # Restore original environment
            if original_env is not None:
                os.environ["LANCEDB_DIR"] = original_env
            else:
                os.environ.pop("LANCEDB_DIR", None)

    @pytest.fixture
    def test_collection(self):
        """Test collection name."""
        return f"test_collection_{uuid.uuid4().hex[:8]}"

    def test_read_chunks_no_data(self, temp_lancedb_dir, test_collection):
        """Test reading chunks when no data exists."""
        result = read_chunks_for_embedding(
            collection=test_collection,
            doc_id="nonexistent_doc",
            parse_hash="nonexistent_hash",
            model="test_model",
        )

        assert isinstance(result, EmbeddingReadResponse)
        assert len(result.chunks) == 0
        assert result.total_count == 0
        assert result.pending_count == 0

    def test_read_chunks_for_embedding_sql_injection_protection(
        self, temp_lancedb_dir, test_collection
    ):
        """Test read_chunks_for_embedding protects against SQL injection."""
        from unittest.mock import MagicMock

        # Create mock vector store
        mock_vector_store = MagicMock()

        # Mock count_rows_or_zero to return 0 (no chunks found)
        mock_vector_store.count_rows_or_zero.return_value = 0

        # Mock iter_batches to return empty batches
        mock_vector_store.iter_batches.return_value = []

        with patch(
            "xagent.core.tools.core.RAG_tools.vector_storage.vector_manager.get_vector_index_store",
            return_value=mock_vector_store,
        ):
            malicious_input = "malicious' OR 1=1 --"
            safe_collection = test_collection
            safe_parse_hash = "safe_hash"
            safe_model = "test_model"

            result = read_chunks_for_embedding(
                collection=safe_collection,
                doc_id=malicious_input,
                parse_hash=safe_parse_hash,
                model=safe_model,
                user_id=None,
                is_admin=True,  # Use admin to avoid user_id filter
            )

            # Verify count_rows_or_zero was called on vector store
            mock_vector_store.count_rows_or_zero.assert_called_once()
            call_kwargs = mock_vector_store.count_rows_or_zero.call_args[1]
            assert call_kwargs["table_name"] == "chunks"
            # Verify filters were passed correctly (including the malicious input)
            assert "collection" in call_kwargs["filters"]
            assert call_kwargs["filters"]["doc_id"] == malicious_input
            assert call_kwargs["filters"]["parse_hash"] == safe_parse_hash

            # Since count_rows_or_zero returns 0, iter_batches should not be called
            mock_vector_store.iter_batches.assert_not_called()

            assert result.chunks == []
            assert result.total_count == 0
            assert result.pending_count == 0


class TestGroupEmbeddingsByModel:
    """Tests for _group_embeddings_by_model helper."""

    def test_group_embeddings_by_model_empty(self):
        """Test grouping empty list returns empty dict."""
        assert _group_embeddings_by_model([]) == {}

    def test_group_embeddings_by_model_single_model(self):
        """Test grouping single model returns one key with all items."""
        from xagent.core.tools.core.RAG_tools.core.schemas import ChunkEmbeddingData

        embeddings = [
            ChunkEmbeddingData(
                collection="c",
                doc_id="d1",
                chunk_id="ch1",
                parse_hash="h",
                model="m1",
                vector=[0.1, 0.2],
                text="t1",
                chunk_hash="ch",
            ),
            ChunkEmbeddingData(
                collection="c",
                doc_id="d2",
                chunk_id="ch2",
                parse_hash="h",
                model="m1",
                vector=[0.2, 0.3],
                text="t2",
                chunk_hash="ch",
            ),
        ]
        result = _group_embeddings_by_model(embeddings)
        assert list(result.keys()) == ["m1"]
        assert len(result["m1"]) == 2

    def test_group_embeddings_by_model_multiple_models(self):
        """Test grouping multiple models returns separate lists."""
        from xagent.core.tools.core.RAG_tools.core.schemas import ChunkEmbeddingData

        embeddings = [
            ChunkEmbeddingData(
                collection="c",
                doc_id="d1",
                chunk_id="ch1",
                parse_hash="h",
                model="m1",
                vector=[0.1, 0.2],
                text="t1",
                chunk_hash="ch",
            ),
            ChunkEmbeddingData(
                collection="c",
                doc_id="d2",
                chunk_id="ch2",
                parse_hash="h",
                model="m2",
                vector=[0.2, 0.3],
                text="t2",
                chunk_hash="ch",
            ),
        ]
        result = _group_embeddings_by_model(embeddings)
        assert set(result.keys()) == {"m1", "m2"}
        assert len(result["m1"]) == 1 and result["m1"][0].model == "m1"
        assert len(result["m2"]) == 1 and result["m2"][0].model == "m2"


class TestValidateAndPrepareTable:
    """Tests for _validate_and_prepare_table helper."""

    def test_validate_and_prepare_table_existing_same_dimension(self):
        """Test table exists with same vector dimension is not dropped."""
        from unittest.mock import MagicMock, patch

        conn = MagicMock(spec=["table_names", "open_table", "drop_table"])
        table_name = "embeddings_test_tag"
        conn.table_names.return_value = [table_name]
        existing_table = MagicMock()
        mock_vector_field = MagicMock()
        mock_vector_field.type.list_size = 2
        mock_schema = MagicMock()
        mock_schema.field.return_value = mock_vector_field
        existing_table.schema = mock_schema
        conn.open_table.return_value = existing_table

        with patch(
            "xagent.core.tools.core.RAG_tools.vector_storage.vector_manager.ensure_embeddings_table"
        ) as mock_ensure:
            result = _validate_and_prepare_table(
                conn, "test_tag", table_name, vector_dim=2
            )
        # Same dimension: should not drop; ensure_embeddings_table then open_table
        conn.drop_table.assert_not_called()
        mock_ensure.assert_called_once_with(conn, "test_tag", vector_dim=2)
        assert result is existing_table

    def test_validate_and_prepare_table_incompatible_vector_type_no_list_size(
        self,
    ):
        """Test table with vector field without list_size is dropped and recreated."""
        from unittest.mock import MagicMock, patch

        conn = MagicMock(spec=["table_names", "open_table", "drop_table"])
        table_name = "embeddings_test_tag"
        conn.table_names.return_value = [table_name]
        existing_table = MagicMock()
        # Use a type object without list_size so hasattr(..., "list_size") is False
        vector_type_no_list_size = type("VectorType", (), {})()
        mock_vector_field = MagicMock()
        mock_vector_field.type = vector_type_no_list_size
        mock_schema = MagicMock()
        mock_schema.field.return_value = mock_vector_field
        existing_table.schema = mock_schema
        conn.open_table.return_value = existing_table
        new_table = MagicMock()
        conn.open_table.side_effect = [existing_table, new_table]

        with patch(
            "xagent.core.tools.core.RAG_tools.vector_storage.vector_manager.ensure_embeddings_table"
        ):
            result = _validate_and_prepare_table(
                conn, "test_tag", table_name, vector_dim=2
            )
        conn.drop_table.assert_called_once_with(table_name)
        assert result is new_table

    def test_validate_and_prepare_table_schema_check_exception_then_recreate(
        self,
    ):
        """Test when schema check raises, drop is attempted and table is recreated."""
        from unittest.mock import MagicMock, patch

        conn = MagicMock(spec=["table_names", "open_table", "drop_table"])
        table_name = "embeddings_test_tag"
        conn.table_names.return_value = [table_name]
        new_table = MagicMock()
        # First open_table (in try) raises; after ensure_embeddings_table, second open_table returns new_table
        conn.open_table.side_effect = [RuntimeError("schema error"), new_table]

        with patch(
            "xagent.core.tools.core.RAG_tools.vector_storage.vector_manager.ensure_embeddings_table"
        ):
            result = _validate_and_prepare_table(
                conn, "test_tag", table_name, vector_dim=2
            )
        conn.drop_table.assert_called_once_with(table_name)
        assert result is new_table


class TestWriteVectorsToDb:
    """Test cases for write_vectors_to_db functionality."""

    @pytest.fixture
    def temp_lancedb_dir(self):
        """Create a temporary directory for LanceDB."""
        with tempfile.TemporaryDirectory() as temp_dir:
            original_env = os.environ.get("LANCEDB_DIR")
            os.environ["LANCEDB_DIR"] = temp_dir
            yield temp_dir
            if original_env is not None:
                os.environ["LANCEDB_DIR"] = original_env
            else:
                os.environ.pop("LANCEDB_DIR", None)

    @pytest.fixture
    def test_collection(self):
        """Test collection name."""
        return f"test_collection_{uuid.uuid4().hex[:8]}"

    def test_write_vectors_empty_list(self, temp_lancedb_dir, test_collection):
        """Test writing empty embedding list."""
        result = write_vectors_to_db(
            collection=test_collection,
            embeddings=[],
        )

        assert isinstance(result, EmbeddingWriteResponse)
        assert result.upsert_count == 0
        assert result.deleted_stale_count == 0
        assert result.index_status == "skipped"

    def test_write_vectors_to_db_sql_injection_protection(
        self, temp_lancedb_dir, test_collection
    ):
        """Test write_vectors_to_db protects against SQL injection."""
        from unittest.mock import MagicMock

        from xagent.core.tools.core.RAG_tools.core.schemas import ChunkEmbeddingData

        # Create mock vector store
        mock_vector_store = MagicMock()
        mock_vector_store.upsert_embeddings.return_value = None
        from xagent.core.tools.core.RAG_tools.core.schemas import IndexResult

        mock_vector_store.create_index.return_value = IndexResult(
            status="below_threshold",
            advice=None,
            fts_enabled=False,
        )

        with patch(
            "xagent.core.tools.core.RAG_tools.vector_storage.vector_manager.get_vector_index_store",
            return_value=mock_vector_store,
        ):
            malicious_doc_id = "malicious' OR 1=1 --"
            safe_collection = test_collection
            safe_parse_hash = "safe_hash"
            safe_model = "test_model"
            malicious_chunk_id = "chunk'id"
            safe_chunk_hash = "safe_hash"

            # Create an embedding with malicious doc_id
            malicious_embedding = ChunkEmbeddingData(
                collection=safe_collection,
                doc_id=malicious_doc_id,
                chunk_id=malicious_chunk_id,
                parse_hash=safe_parse_hash,
                model=safe_model,
                vector=[0.1, 0.2],
                text="malicious text",
                chunk_hash=safe_chunk_hash,
            )

            result = write_vectors_to_db(
                collection=safe_collection,
                embeddings=[malicious_embedding],
            )

            # Verify upsert_embeddings was called on vector store
            mock_vector_store.upsert_embeddings.assert_called_once()
            call_args = mock_vector_store.upsert_embeddings.call_args
            records_arg = call_args[0][1]

            # Verify the records contain the malicious input (properly escaped by LanceDB)
            assert len(records_arg) == 1
            assert records_arg[0]["doc_id"] == malicious_doc_id
            assert records_arg[0]["chunk_id"] == malicious_chunk_id
            assert records_arg[0]["collection"] == safe_collection

            assert result.upsert_count == 1
            assert result.deleted_stale_count == 0
            assert result.index_status == "skipped_threshold"

    def test_write_vectors_merge_insert_fallback_to_add(
        self, temp_lancedb_dir, test_collection
    ):
        """Test merge_insert failure fallback to add method.

        NOTE: This test has been simplified for Phase 1A.
        The actual merge_insert -> add() fallback logic is now implemented
        in LanceDBVectorIndexStore.upsert_embeddings() and should be
        tested in test_lancedb_stores.py. This test only verifies that
        vector_store.upsert_embeddings is called correctly.
        """
        from unittest.mock import MagicMock

        from xagent.core.tools.core.RAG_tools.core.schemas import ChunkEmbeddingData

        # Create mock vector store
        mock_vector_store = MagicMock()
        mock_vector_store.upsert_embeddings.return_value = None
        from xagent.core.tools.core.RAG_tools.core.schemas import IndexResult

        mock_vector_store.create_index.return_value = IndexResult(
            status="below_threshold",
            advice=None,
            fts_enabled=False,
        )

        with patch(
            "xagent.core.tools.core.RAG_tools.vector_storage.vector_manager.get_vector_index_store",
            return_value=mock_vector_store,
        ):
            embedding = ChunkEmbeddingData(
                collection=test_collection,
                doc_id="test_doc",
                chunk_id="test_chunk",
                parse_hash="test_parse",
                model="test_model",
                vector=[0.1, 0.2],
                text="test text",
                chunk_hash="test_hash",
            )

            result = write_vectors_to_db(
                collection=test_collection,
                embeddings=[embedding],
            )

            # Verify upsert_embeddings was called on vector store
            mock_vector_store.upsert_embeddings.assert_called_once()
            assert result.upsert_count == 1

    def test_write_vectors_merge_insert_non_recoverable_error_no_fallback(
        self, temp_lancedb_dir, test_collection
    ):
        """Test that non-recoverable errors propagate correctly.

        NOTE: This test has been simplified for Phase 1A.
        Non-recoverable error handling is now implemented in
        LanceDBVectorIndexStore.upsert_embeddings() and should be
        tested in test_lancedb_stores.py. This test only verifies
        that errors propagate correctly through vector_manager.
        """
        from unittest.mock import MagicMock, patch

        from xagent.core.tools.core.RAG_tools.core.exceptions import (
            DatabaseOperationError,
        )
        from xagent.core.tools.core.RAG_tools.core.schemas import ChunkEmbeddingData

        # Create mock vector store that raises error
        mock_vector_store = MagicMock()
        mock_vector_store.upsert_embeddings.side_effect = ValueError(
            "Schema mismatch: expected int, got string"
        )

        with patch(
            "xagent.core.tools.core.RAG_tools.vector_storage.vector_manager.get_vector_index_store",
            return_value=mock_vector_store,
        ):
            embedding = ChunkEmbeddingData(
                collection=test_collection,
                doc_id="test_doc",
                chunk_id="test_chunk",
                parse_hash="test_parse",
                model="test_model",
                vector=[0.1, 0.2],
                text="test text",
                chunk_hash="test_hash",
            )

            # ValueError is wrapped in DatabaseOperationError by outer exception handler
            with pytest.raises(
                DatabaseOperationError, match="Failed to write embeddings"
            ):
                write_vectors_to_db(
                    collection=test_collection,
                    embeddings=[embedding],
                )

            # Verify upsert_embeddings was called
            mock_vector_store.upsert_embeddings.assert_called_once()

    def test_write_vectors_merge_insert_type_mismatch_error_no_fallback(
        self, temp_lancedb_dir, test_collection
    ):
        """Test that type mismatch errors do not fallback to add (Phase 1A: using storage abstraction)."""
        from unittest.mock import MagicMock, patch

        from xagent.core.tools.core.RAG_tools.core.exceptions import (
            DatabaseOperationError,
        )
        from xagent.core.tools.core.RAG_tools.core.schemas import ChunkEmbeddingData

        # Create mock vector store
        mock_vector_store = MagicMock()

        # Mock upsert_embeddings to fail with type error (non-recoverable)
        mock_vector_store.upsert_embeddings.side_effect = TypeError(
            "Type mismatch: invalid type for field"
        )
        from xagent.core.tools.core.RAG_tools.core.schemas import IndexResult

        mock_vector_store.create_index.return_value = IndexResult(
            status="below_threshold",
            advice=None,
            fts_enabled=False,
        )

        with patch(
            "xagent.core.tools.core.RAG_tools.vector_storage.vector_manager.get_vector_index_store",
            return_value=mock_vector_store,
        ):
            embedding = ChunkEmbeddingData(
                collection=test_collection,
                doc_id="test_doc",
                chunk_id="test_chunk",
                parse_hash="test_parse",
                model="test_model",
                vector=[0.1, 0.2],
                text="test text",
                chunk_hash="test_hash",
            )

            # TypeError is wrapped in DatabaseOperationError by outer exception handler
            with pytest.raises(
                DatabaseOperationError, match="Failed to write embeddings"
            ):
                write_vectors_to_db(
                    collection=test_collection,
                    embeddings=[embedding],
                )

            # Verify upsert_embeddings was called
            mock_vector_store.upsert_embeddings.assert_called_once()

    def test_write_vectors_merge_insert_dimension_error_no_fallback(
        self, temp_lancedb_dir, test_collection
    ):
        """Test that dimension mismatch errors do not fallback to add (Phase 1A: using storage abstraction)."""
        from unittest.mock import MagicMock, patch

        from xagent.core.tools.core.RAG_tools.core.exceptions import (
            DatabaseOperationError,
        )
        from xagent.core.tools.core.RAG_tools.core.schemas import ChunkEmbeddingData

        # Create mock vector store
        mock_vector_store = MagicMock()

        # Mock upsert_embeddings to fail with dimension error (non-recoverable)
        mock_vector_store.upsert_embeddings.side_effect = ValueError(
            "Vector dimension mismatch: expected 3, got 2"
        )
        from xagent.core.tools.core.RAG_tools.core.schemas import IndexResult

        mock_vector_store.create_index.return_value = IndexResult(
            status="below_threshold",
            advice=None,
            fts_enabled=False,
        )

        with patch(
            "xagent.core.tools.core.RAG_tools.vector_storage.vector_manager.get_vector_index_store",
            return_value=mock_vector_store,
        ):
            embedding = ChunkEmbeddingData(
                collection=test_collection,
                doc_id="test_doc",
                chunk_id="test_chunk",
                parse_hash="test_parse",
                model="test_model",
                vector=[0.1, 0.2],
                text="test text",
                chunk_hash="test_hash",
            )

            # ValueError is wrapped in DatabaseOperationError by outer exception handler
            with pytest.raises(
                DatabaseOperationError, match="Failed to write embeddings"
            ):
                write_vectors_to_db(
                    collection=test_collection,
                    embeddings=[embedding],
                )

            # Verify upsert_embeddings was called
            mock_vector_store.upsert_embeddings.assert_called_once()

    def test_write_vectors_merge_insert_recoverable_error_with_fallback(
        self, temp_lancedb_dir, test_collection
    ):
        """Test that recoverable errors (network, timeout) do fallback to add (Phase 1A: using storage abstraction)."""
        from unittest.mock import MagicMock, patch

        from xagent.core.tools.core.RAG_tools.core.schemas import ChunkEmbeddingData

        # Create mock vector store
        mock_vector_store = MagicMock()

        # Mock upsert_embeddings to succeed (it handles fallback internally)
        mock_vector_store.upsert_embeddings.return_value = None
        from xagent.core.tools.core.RAG_tools.core.schemas import IndexResult

        mock_vector_store.create_index.return_value = IndexResult(
            status="below_threshold",
            advice=None,
            fts_enabled=False,
        )

        with patch(
            "xagent.core.tools.core.RAG_tools.vector_storage.vector_manager.get_vector_index_store",
            return_value=mock_vector_store,
        ):
            embedding = ChunkEmbeddingData(
                collection=test_collection,
                doc_id="test_doc",
                chunk_id="test_chunk",
                parse_hash="test_parse",
                model="test_model",
                vector=[0.1, 0.2],
                text="test text",
                chunk_hash="test_hash",
            )

            result = write_vectors_to_db(
                collection=test_collection,
                embeddings=[embedding],
            )

            # Verify upsert_embeddings was called
            mock_vector_store.upsert_embeddings.assert_called_once()
            assert result.upsert_count == 1

    def test_write_vectors_merge_insert_and_add_both_fail(
        self, temp_lancedb_dir, test_collection
    ):
        """Test when both merge_insert and add fail (Phase 1A: using storage abstraction)."""
        from unittest.mock import MagicMock, patch

        from xagent.core.tools.core.RAG_tools.core.exceptions import (
            DatabaseOperationError,
        )
        from xagent.core.tools.core.RAG_tools.core.schemas import ChunkEmbeddingData

        # Create mock vector store
        mock_vector_store = MagicMock()

        # Mock upsert_embeddings to fail
        mock_vector_store.upsert_embeddings.side_effect = Exception("upsert failed")

        with patch(
            "xagent.core.tools.core.RAG_tools.vector_storage.vector_manager.get_vector_index_store",
            return_value=mock_vector_store,
        ):
            embedding = ChunkEmbeddingData(
                collection=test_collection,
                doc_id="test_doc",
                chunk_id="test_chunk",
                parse_hash="test_parse",
                model="test_model",
                vector=[0.1, 0.2],
                text="test text",
                chunk_hash="test_hash",
            )

            with pytest.raises(DatabaseOperationError):
                write_vectors_to_db(
                    collection=test_collection,
                    embeddings=[embedding],
                )

            # Verify upsert_embeddings was called
            mock_vector_store.upsert_embeddings.assert_called_once()

    def test_write_vectors_spill_retry(self, temp_lancedb_dir, test_collection):
        """Test that spill error reduces batch size and retries without losing data (Phase 1A: using storage abstraction)."""
        from unittest.mock import MagicMock, patch

        from xagent.core.tools.core.RAG_tools.core.schemas import ChunkEmbeddingData

        # Create mock vector store
        mock_vector_store = MagicMock()

        # Mock upsert_embeddings to succeed (it handles spill retry internally)
        mock_vector_store.upsert_embeddings.return_value = None
        from xagent.core.tools.core.RAG_tools.core.schemas import IndexResult

        mock_vector_store.create_index.return_value = IndexResult(
            status="below_threshold",
            advice=None,
            fts_enabled=False,
        )

        embeddings = [
            ChunkEmbeddingData(
                collection=test_collection,
                doc_id=f"doc_{i}",
                chunk_id=f"chunk_{i}",
                parse_hash="test_parse",
                model="test_model",
                vector=[0.1, 0.2],
                text=f"text_{i}",
                chunk_hash="test_hash",
            )
            for i in range(5)
        ]

        with (
            patch(
                "xagent.core.tools.core.RAG_tools.vector_storage.vector_manager.get_vector_index_store",
                return_value=mock_vector_store,
            ),
            patch.dict(os.environ, {"LANCEDB_BATCH_SIZE": "2"}, clear=False),
        ):
            result = write_vectors_to_db(
                collection=test_collection,
                embeddings=embeddings,
                create_index=False,
            )

        assert result.upsert_count == 5
        # Verify upsert_embeddings was called 3 times (5 records with batch_size=2)
        # Batch 1: doc_0, doc_1; Batch 2: doc_2, doc_3; Batch 3: doc_4
        assert mock_vector_store.upsert_embeddings.call_count == 3

    def test_write_vectors_batch_partial_failure(
        self, temp_lancedb_dir, test_collection
    ):
        """Test batch processing with partial failures."""
        from unittest.mock import MagicMock, patch

        from xagent.core.tools.core.RAG_tools.core.schemas import ChunkEmbeddingData

        mock_db_connection = MagicMock(
            spec=["table_names", "open_table", "create_table"]
        )
        mock_embeddings_table = _create_mock_table_with_schema()

        def mock_open_table_func(table_name):
            if table_name.startswith("embeddings_"):
                return mock_embeddings_table
            return _create_mock_table_with_schema()

        mock_db_connection.open_table.side_effect = mock_open_table_func
        mock_db_connection.create_table.return_value = None
        mock_db_connection.table_names.return_value = []

        # Create multiple embeddings to trigger batch processing
        embeddings = [
            ChunkEmbeddingData(
                collection=test_collection,
                doc_id=f"doc_{i}",
                chunk_id=f"chunk_{i}",
                parse_hash="test_parse",
                model="test_model",
                vector=[0.1, 0.2],
                text=f"text_{i}",
                chunk_hash="test_hash",
            )
            for i in range(5)
        ]

        # Mock merge_insert to fail for first batch, succeed for others
        call_count = 0

        def mock_merge_insert_side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            mock_merge_insert = MagicMock()
            mock_when_matched = MagicMock()
            mock_when_not_matched = MagicMock()
            mock_merge_insert.when_matched_update_all.return_value = mock_when_matched
            mock_when_matched.when_not_matched_insert_all.return_value = (
                mock_when_not_matched
            )
            if call_count == 1:
                # First batch fails
                mock_when_not_matched.execute.side_effect = Exception("Batch 1 failed")
            else:
                # Other batches succeed
                mock_when_not_matched.execute.return_value = None
            return mock_merge_insert

        mock_embeddings_table.merge_insert.side_effect = mock_merge_insert_side_effect
        # Create mock vector store that uses our mock connection/table
        mock_vector_store = MagicMock()

        def mock_upsert_side_effect(model_tag, records):
            # Simulate real upsert behavior by calling merge_insert on our mock table
            mock_embeddings_table.merge_insert(
                ["collection", "doc_id", "parse_hash", "chunk_id"]
            ).when_matched_update_all().when_not_matched_insert_all().execute(records)

        mock_vector_store.upsert_embeddings.side_effect = mock_upsert_side_effect

        with (
            patch(
                "xagent.core.tools.core.RAG_tools.vector_storage.vector_manager.get_vector_index_store",
                return_value=mock_vector_store,
            ),
            patch.dict(os.environ, {"LANCEDB_BATCH_SIZE": "2"}),
        ):  # Small batch size
            # Now we expect it to raise DatabaseOperationError instead of partial success
            from xagent.core.tools.core.RAG_tools.core.exceptions import (
                DatabaseOperationError,
            )

            with pytest.raises(DatabaseOperationError, match="Batch 1 failed"):
                write_vectors_to_db(
                    collection=test_collection,
                    embeddings=embeddings,
                )

    def test_write_vectors_spill_error_reduces_batch_size(
        self, temp_lancedb_dir, test_collection
    ):
        """Test LanceDB spill error triggers batch size reduction (Phase 1A: using storage abstraction)."""
        from unittest.mock import MagicMock, patch

        from xagent.core.tools.core.RAG_tools.core.schemas import ChunkEmbeddingData

        # Create mock vector store
        mock_vector_store = MagicMock()

        # Mock upsert_embeddings to succeed (it handles spill retry internally)
        mock_vector_store.upsert_embeddings.return_value = None
        from xagent.core.tools.core.RAG_tools.core.schemas import IndexResult

        mock_vector_store.create_index.return_value = IndexResult(
            status="below_threshold",
            advice=None,
            fts_enabled=False,
        )

        # Create embeddings to trigger batch processing
        embeddings = [
            ChunkEmbeddingData(
                collection=test_collection,
                doc_id=f"doc_{i}",
                chunk_id=f"chunk_{i}",
                parse_hash="test_parse",
                model="test_model",
                vector=[0.1, 0.2],
                text=f"text_{i}",
                chunk_hash="test_hash",
            )
            for i in range(5)
        ]

        with (
            patch(
                "xagent.core.tools.core.RAG_tools.vector_storage.vector_manager.get_vector_index_store",
                return_value=mock_vector_store,
            ),
            patch.dict(os.environ, {"LANCEDB_BATCH_SIZE": "100"}),
        ):  # Large batch size
            # Should handle spill error gracefully
            result = write_vectors_to_db(
                collection=test_collection,
                embeddings=embeddings,
            )

            # Verify upsert_embeddings was called
            mock_vector_store.upsert_embeddings.assert_called_once()
            assert result.upsert_count == 5

    def test_write_vectors_schema_mismatch_drops_table(
        self, temp_lancedb_dir, test_collection
    ):
        """Test schema compatibility check and table dropping (Phase 1A: using storage abstraction)."""
        from unittest.mock import MagicMock, patch

        from xagent.core.tools.core.RAG_tools.core.schemas import ChunkEmbeddingData

        # Create mock vector store
        mock_vector_store = MagicMock()

        # Mock upsert_embeddings to succeed (it handles schema mismatch internally)
        mock_vector_store.upsert_embeddings.return_value = None
        from xagent.core.tools.core.RAG_tools.core.schemas import IndexResult

        mock_vector_store.create_index.return_value = IndexResult(
            status="below_threshold",
            advice=None,
            fts_enabled=False,
        )

        with patch(
            "xagent.core.tools.core.RAG_tools.vector_storage.vector_manager.get_vector_index_store",
            return_value=mock_vector_store,
        ):
            embedding = ChunkEmbeddingData(
                collection=test_collection,
                doc_id="test_doc",
                chunk_id="test_chunk",
                parse_hash="test_parse",
                model="test_model",
                vector=[0.1, 0.2],  # 2 dimensions
                text="test text",
                chunk_hash="test_hash",
            )

            result = write_vectors_to_db(
                collection=test_collection,
                embeddings=[embedding],
            )

            # Verify upsert_embeddings was called
            mock_vector_store.upsert_embeddings.assert_called_once()
            assert result.upsert_count == 1

    def test_write_vectors_inconsistent_dimensions(
        self, temp_lancedb_dir, test_collection
    ):
        """Test vector dimension inconsistency detection."""
        from unittest.mock import patch

        from xagent.core.tools.core.RAG_tools.core.exceptions import (
            VectorValidationError,
        )
        from xagent.core.tools.core.RAG_tools.core.schemas import ChunkEmbeddingData

        embeddings = [
            ChunkEmbeddingData(
                collection=test_collection,
                doc_id="doc_1",
                chunk_id="chunk_1",
                parse_hash="test_parse",
                model="test_model",
                vector=[0.1, 0.2],  # 2 dimensions
                text="text_1",
                chunk_hash="test_hash",
            ),
            ChunkEmbeddingData(
                collection=test_collection,
                doc_id="doc_2",
                chunk_id="chunk_2",
                parse_hash="test_parse",
                model="test_model",
                vector=[0.1, 0.2, 0.3],  # 3 dimensions - inconsistent!
                text="text_2",
                chunk_hash="test_hash",
            ),
        ]

        with patch(
            "xagent.core.tools.core.RAG_tools.vector_storage.vector_manager.get_vector_index_store"
        ):
            with pytest.raises(
                VectorValidationError, match="Multiple vector dimensions found"
            ):
                write_vectors_to_db(
                    collection=test_collection,
                    embeddings=embeddings,
                )

    def test_write_vectors_index_creation_failure(
        self, temp_lancedb_dir, test_collection
    ):
        """Test index creation failure handling (Phase 1A: using storage abstraction)."""
        from unittest.mock import MagicMock, patch

        from xagent.core.tools.core.RAG_tools.core.schemas import ChunkEmbeddingData

        # Create mock vector store
        mock_vector_store = MagicMock()

        # Mock upsert_embeddings to succeed
        mock_vector_store.upsert_embeddings.return_value = None
        # Mock create_index to fail
        mock_vector_store.create_index.side_effect = Exception("Index creation failed")

        with patch(
            "xagent.core.tools.core.RAG_tools.vector_storage.vector_manager.get_vector_index_store",
            return_value=mock_vector_store,
        ):
            embedding = ChunkEmbeddingData(
                collection=test_collection,
                doc_id="test_doc",
                chunk_id="test_chunk",
                parse_hash="test_parse",
                model="test_model",
                vector=[0.1, 0.2],
                text="test text",
                chunk_hash="test_hash",
            )

            # Index creation failure should not prevent upsert
            result = write_vectors_to_db(
                collection=test_collection,
                embeddings=[embedding],
            )

            # Verify upsert_embeddings was called
            mock_vector_store.upsert_embeddings.assert_called_once()
            # Verify create_index was called
            mock_vector_store.create_index.assert_called_once()
            # Upsert should succeed even if index creation fails
            assert result.upsert_count == 1

    def test_write_vectors_empty_collection_name(self, temp_lancedb_dir):
        """Test empty collection name validation."""
        from unittest.mock import patch

        from xagent.core.tools.core.RAG_tools.core.exceptions import (
            DocumentValidationError,
        )
        from xagent.core.tools.core.RAG_tools.core.schemas import ChunkEmbeddingData

        embedding = ChunkEmbeddingData(
            collection="",
            doc_id="test_doc",
            chunk_id="test_chunk",
            parse_hash="test_parse",
            model="test_model",
            vector=[0.1, 0.2],
            text="test text",
            chunk_hash="test_hash",
        )

        with patch(
            "xagent.core.tools.core.RAG_tools.vector_storage.vector_manager.get_vector_index_store"
        ):
            with pytest.raises(
                DocumentValidationError, match="Collection name is required"
            ):
                write_vectors_to_db(
                    collection="",
                    embeddings=[embedding],
                )

    def test_write_vectors_multiple_models(self, temp_lancedb_dir, test_collection):
        """Test processing multiple models separately (Phase 1A: using storage abstraction)."""
        from unittest.mock import MagicMock, patch

        from xagent.core.tools.core.RAG_tools.core.schemas import ChunkEmbeddingData

        # Create mock vector store
        mock_vector_store = MagicMock()

        # Mock upsert_embeddings to succeed
        mock_vector_store.upsert_embeddings.return_value = None
        from xagent.core.tools.core.RAG_tools.core.schemas import IndexResult

        mock_vector_store.create_index.return_value = IndexResult(
            status="below_threshold",
            advice=None,
            fts_enabled=False,
        )

        embeddings = [
            ChunkEmbeddingData(
                collection=test_collection,
                doc_id="doc_1",
                chunk_id="chunk_1",
                parse_hash="test_parse",
                model="model_1",
                vector=[0.1, 0.2],
                text="text_1",
                chunk_hash="test_hash",
            ),
            ChunkEmbeddingData(
                collection=test_collection,
                doc_id="doc_2",
                chunk_id="chunk_2",
                parse_hash="test_parse",
                model="model_2",
                vector=[0.3, 0.4],
                text="text_2",
                chunk_hash="test_hash",
            ),
        ]

        with patch(
            "xagent.core.tools.core.RAG_tools.vector_storage.vector_manager.get_vector_index_store",
            return_value=mock_vector_store,
        ):
            result = write_vectors_to_db(
                collection=test_collection,
                embeddings=embeddings,
            )

            # Both models should be processed
            assert result.upsert_count == 2
            # Verify upsert_embeddings was called twice (once for each model)
            assert mock_vector_store.upsert_embeddings.call_count == 2

    def test_write_vectors_batch_size_from_env(self, temp_lancedb_dir, test_collection):
        """Test batch size configuration from environment variable (Phase 1A: using storage abstraction)."""
        from unittest.mock import MagicMock, patch

        from xagent.core.tools.core.RAG_tools.core.schemas import ChunkEmbeddingData

        # Create mock vector store
        mock_vector_store = MagicMock()

        # Mock upsert_embeddings to succeed
        mock_vector_store.upsert_embeddings.return_value = None
        from xagent.core.tools.core.RAG_tools.core.schemas import IndexResult

        mock_vector_store.create_index.return_value = IndexResult(
            status="below_threshold",
            advice=None,
            fts_enabled=False,
        )

        # Create enough embeddings to trigger multiple batches
        embeddings = [
            ChunkEmbeddingData(
                collection=test_collection,
                doc_id=f"doc_{i}",
                chunk_id=f"chunk_{i}",
                parse_hash="test_parse",
                model="test_model",
                vector=[0.1, 0.2],
                text=f"text_{i}",
                chunk_hash="test_hash",
            )
            for i in range(5)
        ]

        with (
            patch(
                "xagent.core.tools.core.RAG_tools.vector_storage.vector_manager.get_vector_index_store",
                return_value=mock_vector_store,
            ),
            patch.dict(os.environ, {"LANCEDB_BATCH_SIZE": "2"}),
        ):  # Custom batch size
            result = write_vectors_to_db(
                collection=test_collection,
                embeddings=embeddings,
            )

            # Should process all embeddings
            assert result.upsert_count == 5
            # Verify upsert_embeddings was called 3 times (5 records with batch_size=2)
            assert mock_vector_store.upsert_embeddings.call_count == 3

    def test_write_vectors_index_status_aggregation(
        self, temp_lancedb_dir, test_collection
    ):
        """Test index status aggregation for multiple models (Phase 1A: using storage abstraction)."""
        from unittest.mock import MagicMock, patch

        from xagent.core.tools.core.RAG_tools.core.schemas import ChunkEmbeddingData

        # Create mock vector store
        mock_vector_store = MagicMock()

        # Mock upsert_embeddings to succeed
        mock_vector_store.upsert_embeddings.return_value = None
        # Mock create_index with different statuses for different models
        from xagent.core.tools.core.RAG_tools.core.schemas import IndexResult

        mock_vector_store.create_index.side_effect = [
            IndexResult(
                status="index_building",
                advice=None,
                fts_enabled=False,
            ),  # First model
            IndexResult(
                status="failed",
                advice=None,
                fts_enabled=False,
            ),  # Second model
        ]

        embeddings = [
            ChunkEmbeddingData(
                collection=test_collection,
                doc_id="doc_1",
                chunk_id="chunk_1",
                parse_hash="test_parse",
                model="model_1",
                vector=[0.1, 0.2],
                text="text_1",
                chunk_hash="test_hash",
            ),
            ChunkEmbeddingData(
                collection=test_collection,
                doc_id="doc_2",
                chunk_id="chunk_2",
                parse_hash="test_parse",
                model="model_2",
                vector=[0.3, 0.4],
                text="text_2",
                chunk_hash="test_hash",
            ),
        ]

        with patch(
            "xagent.core.tools.core.RAG_tools.vector_storage.vector_manager.get_vector_index_store",
            return_value=mock_vector_store,
        ):
            result = write_vectors_to_db(
                collection=test_collection,
                embeddings=embeddings,
                create_index=True,
            )

            # Both models should be processed
            assert result.upsert_count == 2
            # Verify upsert_embeddings was called twice (once for each model)
            assert mock_vector_store.upsert_embeddings.call_count == 2
            # Verify create_index was called twice (once for each model)
            assert mock_vector_store.create_index.call_count == 2
            # Overall status should reflect aggregation (index_building takes precedence)
            from xagent.core.tools.core.RAG_tools.core.schemas import IndexOperation

            assert result.index_status == IndexOperation.CREATED.value

            # index_building should take priority over failed
            assert result.index_status == "created"


class TestVectorValidation:
    """Test cases for vector validation functionality."""

    def test_validate_query_vector_valid(self):
        """Test validation of valid query vectors."""
        # Test valid vectors
        validate_query_vector([1.0, 2.0, 3.0])
        validate_query_vector([0.5, -0.5, 0.0])
        validate_query_vector([1, 2, 3])  # integers are valid

    def test_validate_query_vector_invalid_type(self):
        """Test validation with invalid types."""
        with pytest.raises(VectorValidationError, match="query_vector must be a list"):
            validate_query_vector("not a list")

        with pytest.raises(VectorValidationError, match="query_vector must be a list"):
            validate_query_vector(None)

    def test_validate_query_vector_empty(self):
        """Test validation of empty vector."""
        with pytest.raises(VectorValidationError, match="query_vector cannot be empty"):
            validate_query_vector([])

    def test_validate_query_vector_invalid_elements(self):
        """Test validation with invalid vector elements."""
        with pytest.raises(
            VectorValidationError, match="query_vector must contain only numbers"
        ):
            validate_query_vector([1.0, "invalid", 3.0])

        with pytest.raises(
            VectorValidationError, match="query_vector must contain only numbers"
        ):
            validate_query_vector([1.0, None, 3.0])

    def test_validate_query_vector_nan_infinity(self):
        """Test validation with NaN and infinity values."""

        with pytest.raises(
            VectorValidationError, match="query_vector contains invalid values"
        ):
            validate_query_vector([1.0, float("nan"), 3.0])

        with pytest.raises(
            VectorValidationError, match="query_vector contains invalid values"
        ):
            validate_query_vector([1.0, float("inf"), 3.0])

        with pytest.raises(
            VectorValidationError, match="query_vector contains invalid values"
        ):
            validate_query_vector([1.0, -float("inf"), 3.0])

    def test_validate_query_vector_numpy_scalar_types(self):
        """Test validation with numpy scalar types (np.int32, np.float64, etc.)."""
        try:
            import numpy as np

            # Test with numpy scalar types - should pass validation
            validate_query_vector([np.float64(1.0), np.float32(2.0), np.int32(3)])
            validate_query_vector([np.float64(0.5), np.float32(-0.5), np.int64(0)])
            validate_query_vector([np.float64(1.0), 2.0, np.int32(3)])  # Mixed types

            # Test with numpy array elements (should also work)
            validate_query_vector([np.float64(1.0), np.float64(2.0), np.float64(3.0)])

        except ImportError:
            pytest.skip("numpy not available")


class TestValidateQueryVectorExtended:
    """Test extended validate_query_vector functionality with model and dimension validation."""

    @pytest.fixture
    def temp_lancedb_dir(self):
        """Create a temporary directory for LanceDB."""
        with tempfile.TemporaryDirectory() as temp_dir:
            original_env = os.environ.get("LANCEDB_DIR")
            os.environ["LANCEDB_DIR"] = temp_dir
            yield temp_dir
            if original_env is not None:
                os.environ["LANCEDB_DIR"] = original_env
            else:
                os.environ.pop("LANCEDB_DIR", None)

    @pytest.fixture
    def test_collection(self):
        """Test collection name."""
        return f"test_collection_{uuid.uuid4().hex[:8]}"

    def test_validate_without_connection(self):
        """Test validation without database connection (backward compatibility)."""
        # Should work without model_tag and conn parameters
        validate_query_vector([1.0, 2.0, 3.0])

        # Should work with model_tag but no conn
        validate_query_vector([1.0, 2.0, 3.0], model_tag="test_model")


class TestReindexingFunctionality:
    """Test cases for reindexing functionality."""

    @pytest.fixture
    def temp_lancedb_dir(self):
        """Create a temporary directory for LanceDB."""
        with tempfile.TemporaryDirectory() as temp_dir:
            original_env = os.environ.get("LANCEDB_DIR")
            os.environ["LANCEDB_DIR"] = temp_dir
            yield temp_dir
            if original_env is not None:
                os.environ["LANCEDB_DIR"] = original_env
            else:
                os.environ.pop("LANCEDB_DIR", None)

    @pytest.fixture
    def test_collection(self):
        """Test collection name."""
        return f"test_collection_{uuid.uuid4().hex[:8]}"

    def test_write_vectors_with_reindex_integration(
        self, temp_lancedb_dir, test_collection
    ):
        """Test write_vectors_to_db with reindex integration (Phase 1A: using storage abstraction)."""
        from unittest.mock import MagicMock, patch

        from xagent.core.tools.core.RAG_tools.core.schemas import ChunkEmbeddingData

        # Create mock vector store
        mock_vector_store = MagicMock()

        # Mock upsert_embeddings to succeed
        mock_vector_store.upsert_embeddings.return_value = None
        # Mock create_index to return index_building status
        from xagent.core.tools.core.RAG_tools.core.schemas import IndexResult

        mock_vector_store.create_index.return_value = IndexResult(
            status="index_building",
            advice=None,
            fts_enabled=False,
        )

        with patch(
            "xagent.core.tools.core.RAG_tools.vector_storage.vector_manager.get_vector_index_store",
            return_value=mock_vector_store,
        ):
            embedding = ChunkEmbeddingData(
                collection=test_collection,
                doc_id="test_doc",
                chunk_id="test_chunk",
                parse_hash="test_parse",
                model="test_model",
                vector=[0.1, 0.2],
                text="test text",
                chunk_hash="test_hash",
            )

            result = write_vectors_to_db(
                collection=test_collection,
                embeddings=[embedding],
                create_index=True,
            )

            # Verify upsert_embeddings was called
            mock_vector_store.upsert_embeddings.assert_called_once()
            # Verify create_index was called
            mock_vector_store.create_index.assert_called_once()
            assert result.upsert_count == 1
            # Verify index status reflects building state
            from xagent.core.tools.core.RAG_tools.core.schemas import IndexOperation

            assert result.index_status == IndexOperation.CREATED.value

    def test_write_vectors_reindex_policy_configuration(
        self, temp_lancedb_dir, test_collection
    ):
        """Test write_vectors_to_db with different reindex policy configurations (Phase 1A: using storage abstraction)."""
        from unittest.mock import MagicMock, patch

        from xagent.core.tools.core.RAG_tools.core.schemas import ChunkEmbeddingData

        # Create mock vector store
        mock_vector_store = MagicMock()

        # Mock upsert_embeddings to succeed
        mock_vector_store.upsert_embeddings.return_value = None
        # Mock create_index to return index_building status
        from xagent.core.tools.core.RAG_tools.core.schemas import IndexResult

        mock_vector_store.create_index.return_value = IndexResult(
            status="index_building",
            advice=None,
            fts_enabled=False,
        )

        with (
            patch(
                "xagent.core.tools.core.RAG_tools.vector_storage.vector_manager.get_vector_index_store",
                return_value=mock_vector_store,
            ),
        ):
            embedding = ChunkEmbeddingData(
                collection=test_collection,
                doc_id="test_doc",
                chunk_id="test_chunk",
                parse_hash="test_parse",
                model="test_model",
                vector=[0.1, 0.2],
                text="test text",
                chunk_hash="test_hash",
            )

            result = write_vectors_to_db(
                collection=test_collection,
                embeddings=[embedding],
                create_index=True,
            )

            # Verify upsert_embeddings was called
            mock_vector_store.upsert_embeddings.assert_called_once()
            # Verify create_index was called
            mock_vector_store.create_index.assert_called_once()
            assert result.upsert_count == 1

    def test_read_chunks_arrow_fallback_chain(
        self, temp_lancedb_dir, test_collection
    ) -> None:
        """Test read_chunks_for_embedding using storage abstraction (Phase 1A).

        Note: This test now uses the abstraction layer. The original Arrow fallback chain
        (to_arrow → to_list → to_pandas) is handled within LanceDB's iter_batches() implementation.
        """
        from unittest.mock import MagicMock, patch

        # Create mock vector store
        mock_vector_store = MagicMock()

        # Create test chunks data as PyArrow RecordBatch
        import pyarrow as pa

        # Create a proper RecordBatch
        chunks_data = {
            "chunk_id": ["chunk1"],
            "text": ["test content"],
            "collection": [test_collection],
            "doc_id": ["doc1"],
            "parse_hash": ["hash1"],
            "index": [0],
            "chunk_hash": ["test_hash"],
            "metadata": ['{"key": "value"}'],
        }
        mock_batch = pa.RecordBatch.from_pydict(chunks_data)

        # Mock count_rows_or_zero to return 1
        mock_vector_store.count_rows_or_zero.return_value = 1

        # Mock iter_batches to return batches (returns RecordBatch iterator)
        mock_vector_store.iter_batches.return_value = iter([mock_batch])

        with patch(
            "xagent.core.tools.core.RAG_tools.vector_storage.vector_manager.get_vector_index_store",
            return_value=mock_vector_store,
        ):
            result = read_chunks_for_embedding(
                collection=test_collection,
                doc_id="doc1",
                parse_hash="hash1",
                model="test_model",
            )

            assert result.total_count == 1
            assert len(result.chunks) == 1
            # Verify the abstraction methods were called
            # After Phase 1A: count_rows_or_zero and iter_batches called twice (chunks + embeddings tables)
            assert mock_vector_store.count_rows_or_zero.call_count == 2
            assert mock_vector_store.iter_batches.call_count == 2

    @pytest.mark.skip(
        "Legacy fallback test replaced by storage abstraction. "
        "The Arrow → pandas fallback is now handled by LanceDB's iter_batches() "
        "and vector_manager's to_pandas() conversion."
    )
    def test_read_chunks_fallback_to_list(
        self, temp_lancedb_dir, test_collection
    ) -> None:
        """Legacy test - Arrow fallback chain is now handled by LanceDB internals."""

    def test_read_chunks_with_nan_normalization(
        self, temp_lancedb_dir, test_collection
    ) -> None:
        """Test read_chunks_for_embedding with NaN normalization (Phase 1A)."""
        from unittest.mock import MagicMock, patch

        # Create mock vector store
        mock_vector_store = MagicMock()

        # Create test chunks data with NaN (using None for optional fields in PyArrow)
        import pyarrow as pa

        chunks_data = {
            "chunk_id": ["chunk1"],
            "text": ["test content"],
            "collection": [test_collection],
            "doc_id": ["doc1"],
            "parse_hash": ["hash1"],
            "index": [0],
            "chunk_hash": ["test_hash"],
            "metadata": ['{"key": "value"}'],
            "page_number": [None],  # None represents missing/NaN optional field
        }
        mock_batch = pa.RecordBatch.from_pydict(chunks_data)

        # Mock count_rows_or_zero to return 1
        mock_vector_store.count_rows_or_zero.return_value = 1

        # Mock iter_batches to return batches (returns RecordBatch iterator)
        mock_vector_store.iter_batches.return_value = iter([mock_batch])

        with patch(
            "xagent.core.tools.core.RAG_tools.vector_storage.vector_manager.get_vector_index_store",
            return_value=mock_vector_store,
        ):
            result = read_chunks_for_embedding(
                collection=test_collection,
                doc_id="doc1",
                parse_hash="hash1",
                model="test_model",
            )

            assert result.total_count == 1
            assert len(result.chunks) == 1
            # Verify the abstraction methods were called
            # After Phase 1A: count_rows_or_zero and iter_batches called twice (chunks + embeddings tables)
            assert mock_vector_store.count_rows_or_zero.call_count == 2
            assert mock_vector_store.iter_batches.call_count == 2
            # Verify None/NaN was properly handled (page_number should be None)
            assert result.chunks[0].page_number is None
