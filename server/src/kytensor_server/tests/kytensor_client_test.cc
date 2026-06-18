// Copyright 2025. All Rights Reserved.
// Author: Arthur Bin
// Date: 2025-10-16
// Description: kytensor client unit test
#include "../src/kytensor_client.h"
#include <gtest/gtest.h>
#include <thread>
#include <chrono>

namespace kc = kytensor::client;

// Test fixture for KyTensorClient IsServerLive tests
class KyTensorClientIsServerLiveTest : public ::testing::Test {
protected:
    void SetUp() override {
    }

    void TearDown() override {
        // Clean up any resources after each test
    }
};

// Test that IsServerLive returns success when the server is running
TEST_F(KyTensorClientIsServerLiveTest, ServerLiveSuccess) {
    std::unique_ptr<kc::KyTensorClient> client;
    kc::Error err = kc::KyTensorClient::Create(&client);

    if(err.IsOk() == false) {
        std::cerr << "Failed to create KyTensorClient: " << err.Message() << std::endl;
        FAIL();
        return;
    }

    // Give the client some time to connect
    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    // Test IsServerLive functionality
    bool live = false;
    err = client->IsServerLive(&live, 5000); // 5 second timeout

    // Note: In a real test environment with a running server, we would expect:
    EXPECT_TRUE(err.IsOk()) << "IsServerLive should succeed when server is running";
    EXPECT_TRUE(live) << "Server should be live";
}

// Test fixture for KyTensorClient IsServerReady tests
class KyTensorClientIsServerReadyTest : public ::testing::Test {
protected:
    void SetUp() override {
    }

    void TearDown() override {
    }
};

// Test that IsServerReady returns success when the server is running
TEST_F(KyTensorClientIsServerReadyTest, ServerReadySuccess) {
    std::unique_ptr<kc::KyTensorClient> client;
    kc::Error err = kc::KyTensorClient::Create(&client);

    if(err.IsOk() == false) {
        std::cerr << "Failed to create KyTensorClient: " << err.Message() << std::endl;
        FAIL();
        return;
    }

    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    bool ready = false;
    err = client->IsServerReady(&ready, 5000); // 5 second timeout

    EXPECT_TRUE(err.IsOk()) << "IsServerReady should succeed when server is running";
    EXPECT_TRUE(ready) << "Server should be ready";
}

// Test fixture for KyTensorClient IsModelReady tests
class KyTensorClientIsModelReadyTest : public ::testing::Test {
protected:
    void SetUp() override {
    }

    void TearDown() override {
    }
};

// Test that IsModelReady returns success when the model is running
TEST_F(KyTensorClientIsModelReadyTest, ModelReadySuccess) {
    std::unique_ptr<kc::KyTensorClient> client;
    kc::Error err = kc::KyTensorClient::Create(&client);

    if(err.IsOk() == false) {
        std::cerr << "Failed to create KyTensorClient: " << err.Message() << std::endl;
        FAIL();
        return;
    }

    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    bool ready = false;
    err = client->IsModelReady(&ready, "test_model", "", 5000); // 5 second timeout

    EXPECT_TRUE(err.IsOk()) << "IsModelReady should succeed";
}

// Test IsModelReady with specific model version
TEST_F(KyTensorClientIsModelReadyTest, ModelReadyWithVersion) {
    std::unique_ptr<kc::KyTensorClient> client;
    kc::Error err = kc::KyTensorClient::Create(&client);

    if(err.IsOk() == false) {
        std::cerr << "Failed to create KyTensorClient: " << err.Message() << std::endl;
        FAIL();
        return;
    }

    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    bool ready = false;
    err = client->IsModelReady(&ready, "test_model", "1", 5000); // 5 second timeout

    EXPECT_TRUE(err.IsOk()) << "IsModelReady should succeed";
}

// Test fixture for KyTensorClient ServerMetadata tests
class KyTensorClientServerMetadataTest : public ::testing::Test {
protected:
    void SetUp() override {
    }

    void TearDown() override {
    }
};

// Test that ServerMetadata returns success when the server is running
TEST_F(KyTensorClientServerMetadataTest, ServerMetadataSuccess) {
    std::unique_ptr<kc::KyTensorClient> client;
    kc::Error err = kc::KyTensorClient::Create(&client);

    if(err.IsOk() == false) {
        std::cerr << "Failed to create KyTensorClient: " << err.Message() << std::endl;
        FAIL();
        return;
    }

    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    inference::ServerMetadataResponse response;
    err = client->ServerMetadata(&response, 5000); // 5 second timeout

    EXPECT_TRUE(err.IsOk()) << "ServerMetadata should succeed when server is running";
}

// Test fixture for KyTensorClient ModelMetadata tests
class KyTensorClientModelMetadataTest : public ::testing::Test {
protected:
    void SetUp() override {
    }

    void TearDown() override {
    }
};

// Test that ModelMetadata returns success when the model exists
TEST_F(KyTensorClientModelMetadataTest, ModelMetadataSuccess) {
    std::unique_ptr<kc::KyTensorClient> client;
    kc::Error err = kc::KyTensorClient::Create(&client);

    if(err.IsOk() == false) {
        std::cerr << "Failed to create KyTensorClient: " << err.Message() << std::endl;
        FAIL();
        return;
    }

    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    client->LoadModel("simple", "", {}, 5000);

    inference::ModelMetadataResponse response;
    err = client->ModelMetadata(&response, "simple", "", 5000); // 5 second timeout

    client->UnloadModel("simple", 5000);

    EXPECT_TRUE(err.IsOk()) << "ModelMetadata should succeed";
}

// Test ModelMetadata with specific model version
TEST_F(KyTensorClientModelMetadataTest, ModelMetadataWithVersion) {
    std::unique_ptr<kc::KyTensorClient> client;
    kc::Error err = kc::KyTensorClient::Create(&client);

    if(err.IsOk() == false) {
        std::cerr << "Failed to create KyTensorClient: " << err.Message() << std::endl;
        FAIL();
        return;
    }

    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    client->LoadModel("simple", "", {}, 5000);

    inference::ModelMetadataResponse response;
    err = client->ModelMetadata(&response, "simple", "1", 5000); // 5 second timeout

    client->UnloadModel("simple", 5000);

    EXPECT_TRUE(err.IsOk()) << "ModelMetadata should succeed";
}

// Test fixture for KyTensorClient ModelConfig tests
class KyTensorClientModelConfigTest : public ::testing::Test {
protected:
    void SetUp() override {
    }

    void TearDown() override {
    }
};

// Test that ModelConfig returns success when the model exists
TEST_F(KyTensorClientModelConfigTest, ModelConfigSuccess) {
    std::unique_ptr<kc::KyTensorClient> client;
    kc::Error err = kc::KyTensorClient::Create(&client);

    if(err.IsOk() == false) {
        std::cerr << "Failed to create KyTensorClient: " << err.Message() << std::endl;
        FAIL();
        return;
    }

    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    client->LoadModel("simple", "", {}, 5000);

    inference::ModelConfigResponse response;
    err = client->ModelConfig(&response, "simple", "", 5000); // 5 second timeout

    client->UnloadModel("simple", 5000);

    EXPECT_TRUE(err.IsOk()) << "ModelConfig should succeed";
}

// Test ModelConfig with specific model version
TEST_F(KyTensorClientModelConfigTest, ModelConfigWithVersion) {
    std::unique_ptr<kc::KyTensorClient> client;
    kc::Error err = kc::KyTensorClient::Create(&client);

    if(err.IsOk() == false) {
        std::cerr << "Failed to create KyTensorClient: " << err.Message() << std::endl;
        FAIL();
        return;
    }

    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    client->LoadModel("simple", "", {}, 5000);

    inference::ModelConfigResponse response;
    err = client->ModelConfig(&response, "simple", "1", 5000); // 5 second timeout

    client->UnloadModel("simple", 5000);

    EXPECT_TRUE(err.IsOk()) << "ModelConfig should succeed";
}

// Test fixture for KyTensorClient ModelRepositoryIndex tests
class KyTensorClientModelRepositoryIndexTest : public ::testing::Test {
protected:
    void SetUp() override {
    }

    void TearDown() override {
    }
};

// Test that ModelRepositoryIndex returns success when server is running
TEST_F(KyTensorClientModelRepositoryIndexTest, ModelRepositoryIndexSuccess) {
    std::unique_ptr<kc::KyTensorClient> client;
    kc::Error err = kc::KyTensorClient::Create(&client);

    if(err.IsOk() == false) {
        std::cerr << "Failed to create KyTensorClient: " << err.Message() << std::endl;
        FAIL();
        return;
    }

    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    inference::RepositoryIndexResponse response;
    err = client->ModelRepositoryIndex(&response, 5000); // 5 second timeout

    EXPECT_TRUE(err.IsOk()) << "ModelRepositoryIndex should succeed";
}

// Test fixture for KyTensorClient LoadModel and UnloadModel tests
class KyTensorClientModelManagementTest : public ::testing::Test {
protected:
    void SetUp() override {
    }

    void TearDown() override {
    }
};

// Test that LoadModel returns success when the model can be loaded
// ./kytensor --model-repository=/home/kylin/bzm/model_repo1   --model-control-mode=explicit
TEST_F(KyTensorClientModelManagementTest, LoadModelSuccess) {
    std::unique_ptr<kc::KyTensorClient> client;
    kc::Error err = kc::KyTensorClient::Create(&client);

    if(err.IsOk() == false) {
        std::cerr << "Failed to create KyTensorClient: " << err.Message() << std::endl;
        FAIL();
        return;
    }

    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    err = client->LoadModel("simple", "", {}, 5000); // 5 second timeout

    EXPECT_TRUE(err.IsOk()) << "LoadModel should succeed";
}

// Test that UnloadModel returns success when the model can be unloaded
TEST_F(KyTensorClientModelManagementTest, UnloadModelSuccess) {
    std::unique_ptr<kc::KyTensorClient> client;
    kc::Error err = kc::KyTensorClient::Create(&client);

    if(err.IsOk() == false) {
        std::cerr << "Failed to create KyTensorClient: " << err.Message() << std::endl;
        FAIL();
        return;
    }

    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    err = client->LoadModel("simple", "", {}, 5000);

    err = client->UnloadModel("simple", 5000); // 5 second timeout

    EXPECT_TRUE(err.IsOk()) << "UnloadModel should succeed";
}

// Test fixture for KyTensorClient ModelInferenceStatistics tests
class KyTensorClientModelInferenceStatisticsTest : public ::testing::Test {
protected:
    void SetUp() override {
    }

    void TearDown() override {
    }
};

// Test ModelInferenceStatistics with specific model and version
// ModelInferenceStatistics undo in server
TEST_F(KyTensorClientModelInferenceStatisticsTest, ModelInferenceStatisticsWithModel) {
    std::unique_ptr<kc::KyTensorClient> client;
    kc::Error err = kc::KyTensorClient::Create(&client);

    if(err.IsOk() == false) {
        std::cerr << "Failed to create KyTensorClient: " << err.Message() << std::endl;
        FAIL();
        return;
    }

    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    client->LoadModel("simple", "", {}, 5000);

    inference::ModelStatisticsResponse response;
    err = client->LoadModel("simple", "", {}, 5000);
    err = client->ModelInferenceStatistics(&response, "simple", "1", 5000); // 5 second timeout

    EXPECT_TRUE(err.IsOk()) << "ModelInferenceStatistics should succeed";
}

// Test fixture for KyTensorClient Start/Stop functionality
class KyTensorClientStartStopTest : public ::testing::Test {
protected:
    void SetUp() override {
    }

    void TearDown() override {
    }
};

// Test that Start method works correctly
TEST_F(KyTensorClientStartStopTest, StartSuccess) {
    std::unique_ptr<kc::KyTensorClient> client;
    kc::Error err = kc::KyTensorClient::Create(&client);

    EXPECT_TRUE(err.Code() == kc::KYTENSOR_ERROR_SUCCESS) << "Start should return true when connection is successful";
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}