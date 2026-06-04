// Copyright 2014 Manu Martinez-Almeida. All rights reserved.
// Use of this source code is governed by a MIT style
// license that can be found in the LICENSE file.

package gin

import (
	"bytes"
	"context"
	"crypto/tls"
	"errors"
	"fmt"
	"html/template"
	"io"
	"io/fs"
	"mime/multipart"
	"net"
	"net/http"
	"net/http/httptest"
	"net/url"
	"os"
	"path/filepath"
	"reflect"
	"strconv"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/gin-contrib/sse"
	"github.com/gin-gonic/gin/binding"
	"github.com/gin-gonic/gin/codec/json"
	testdata "github.com/gin-gonic/gin/testdata/protoexample"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"go.mongodb.org/mongo-driver/v2/bson"
	"google.golang.org/protobuf/proto"
)

func TestContextSetGet(t *testing.T) {
	c, _ := CreateTestContext(httptest.NewRecorder())
	c.Set("test", "test_value")
	value, err := c.Get("test")
	assert.NoError(t, err)
	assert.Equal(t, "test_value", value)
}

func TestContextSetGetAny(t *testing.T) {
	c, _ := CreateTestContext(httptest.NewRecorder())
	c.SetAny("test", "test_value")
	value, err := c.GetAny("test")
	assert.NoError(t, err)
	assert.Equal(t, "test_value", value)
}

func TestContextSetGetValues(t *testing.T) {
	c, _ := CreateTestContext(httptest.NewRecorder())
	c.Set("string", "this is a string")
	c.Set("int32", int32(-42))
	c.Set("int64", int64(42424242424242))
	c.Set("uint64", uint64(42))
	c.Set("float32", float32(4.2))
	c.Set("float64", 4.2)
	var a any = 1
	c.Set("intInterface", a)

	assert.Exactly(t, "this is a string", c.MustGet("string").(string))
	assert.Exactly(t, int32(-42), c.MustGet("int32").(int32))
	assert.Exactly(t, int64(42424242424242), c.MustGet("int64").(int64))
	assert.Exactly(t, uint64(42), c.MustGet("uint64").(uint64))
	assert.InDelta(t, float32(4.2), c.MustGet("float32").(float32), 0.01)
	assert.InDelta(t, 4.2, c.MustGet("float64").(float64), 0.01)
	assert.Exactly(t, 1, c.MustGet("intInterface").(int))
}

func TestContextGetString(t *testing.T) {
	c, _ := CreateTestContext(httptest.NewRecorder())
	c.Set("string", "this is a string")
	assert.Equal(t, "this is a string", c.GetString("string"))
}

func TestContextSetGetBool(t *testing.T) {
	c, _ := CreateTestContext(httptest.NewRecorder())
	c.Set("bool", true)
	assert.True(t, c.GetBool("bool"))
}

func TestSetGetDelete(t *testing.T) {
	c, _ := CreateTestContext(httptest.NewRecorder())
	key := "example-key"
	value := "example-value"
	c.Set(key, value)
	val, exists := c.Get(key)
	assert.True(t, exists)
	assert.Equal(t, val, value)
	c.Delete(key)
	_, exists = c.Get(key)
	assert.False(t, exists)
}

func TestContextGetInt(t *testing.T) {
	c, _ := CreateTestContext(httptest.NewRecorder())
	c.Set("int", 1)
	assert.Equal(t, 1, c.GetInt("int"))
}

func TestContextGetInt8(t *testing.T) {
	c, _ := CreateTestContext(httptest.NewRecorder())
	key := "int8"
	value := int8(0x7F)
	c.Set(key, value)
	assert.Equal(t, value, c.GetInt8(key))
}

func TestContextGetInt16(t *testing.T) {
	c, _ := CreateTestContext(httptest.NewRecorder())
	key := "int16"
	value := int16(0x7FFF)
	c.Set(key, value)
	assert.Equal(t, value, c.GetInt16(key))
}

func TestContextGetInt32(t *testing.T) {
	c, _ := CreateTestContext(httptest.NewRecorder())
	key := "int32"
	value := int32(0x7FFFFFFF)
	c.Set(key, value)
	assert.Equal(t, value, c.GetInt32(key))
}

func TestContextGetInt64(t *testing.T) {
	c, _ := CreateTestContext(httptest.NewRecorder())
	c.Set("int64", int64(42424242424242))
	assert.Equal(t, int64(42424242424242), c.GetInt64("int64"))
}

func TestContextGetUint(t *testing.T) {
	c, _ := CreateTestContext(httptest.NewRecorder())
	c.Set("uint", uint(1))
	assert.Equal(t, uint(1), c.GetUint("uint"))
}

func TestContextGetUint8(t *testing.T) {
	c, _ := CreateTestContext(httptest.NewRecorder())
	key := "uint8"
	value := uint8(0xFF)
	c.Set(key, value)
	assert.Equal(t, value, c.GetUint8(key))
}

func TestContextGetUint16(t *testing.T) {
	c, _ := CreateTestContext(httptest.NewRecorder())
	key := "uint16"
	value := uint16(0xFFFF)
	c.Set(key, value)
	assert.Equal(t, value, c.GetUint16(key))
}

func TestContextGetUint32(t *testing.T) {
	c, _ := CreateTestContext(httptest.NewRecorder())
	key := "uint32"
	value := uint32(0xFFFFFFFF)
	c.Set(key, value)
	assert.Equal(t, value, c.GetUint32(key))
}

func TestContextGetUint64(t *testing.T) {
	c, _ := CreateTestContext(httptest.NewRecorder())
	c.Set("uint64", uint64(18446744073709551615))
	assert.Equal(t, uint64(18446744073709551615), c.GetUint64("uint64"))
}

func TestContextGetFloat32(t *testing.T) {
	c, _ := CreateTestContext(httptest.NewRecorder())
	key := "float32"
	value := float32(4.2)
	c.Set(key, value)
	assert.Equal(t, value, c.GetFloat32(key))
}

func TestContextSaveUploadedFilePermissions(t *testing.T) {
	getFileHeader := func(t *testing.T) *multipart.FileHeader {
		t.Helper()
		var body bytes.Buffer
		writer := multipart.NewWriter(&body)
		part, err := writer.CreateFormFile("file", "test.txt")
		require.NoError(t, err)
		_, err = part.Write([]byte("hello world"))
		require.NoError(t, err)
		require.NoError(t, writer.Close())

		reader := multipart.NewReader(&body, writer.Boundary())
		form, err := reader.ReadForm(32 << 20)
		require.NoError(t, err)
		t.Cleanup(func() { form.RemoveAll() })

		require.Len(t, form.File["file"], 1)
		return form.File["file"][0]
	}

	t.Run("pre-existing directory", func(t *testing.T) {
		parent := t.TempDir()
		existingDir := filepath.Join(parent, "existing")
		require.NoError(t, os.Mkdir(existingDir, 0o755))

		info, err := os.Stat(existingDir)
		require.NoError(t, err)
		originalMode := info.Mode().Perm()

		fileHeader := getFileHeader(t)
		c, _ := CreateTestContext(httptest.NewRecorder())
		dst := filepath.Join(existingDir, "test.txt")
		err = c.SaveUploadedFile(fileHeader, dst)
		require.NoError(t, err)

		info, err = os.Stat(existingDir)
		require.NoError(t, err)
		assert.Equal(t, originalMode, info.Mode().Perm())

		content, err := os.ReadFile(dst)
		require.NoError(t, err)
		assert.Equal(t, "hello world", string(content))
	})

	t.Run("newly created directory", func(t *testing.T) {
		parent := t.TempDir()
		leafDir := filepath.Join(parent, "new", "leaf")

		fileHeader := getFileHeader(t)
		c, _ := CreateTestContext(httptest.NewRecorder())
		dst := filepath.Join(leafDir, "test.txt")
		err := c.SaveUploadedFile(fileHeader, dst)
		require.NoError(t, err)

		info, err := os.Stat(leafDir)
		require.NoError(t, err)
		assert.Equal(t, os.FileMode(0o750), info.Mode().Perm())

		content, err := os.ReadFile(dst)
		require.NoError(t, err)
		assert.Equal(t, "hello world", string(content))
	})
}
